import json
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a supply chain analyst assistant for a sustainable packaging company.
You explain demand forecasts to operations teams in clear, actionable language.
Be concise. Use plain English. No jargon. Always end with one specific recommendation."""


def build_forecast_context(summary: dict, category: str, store: str) -> str:
    """Convert forecast summary dict into a readable context string for the LLM."""
    lines = [f"Demand forecast for category '{category}' at store '{store}':\n"]
    for item_id, stats in summary.items():
        direction = "up" if stats["pct_change"] > 0 else "down"
        lines.append(
            f"- {item_id}: avg daily demand {direction} {abs(stats['pct_change'])}% "
            f"(prior {stats['prior_avg_daily']} → predicted {stats['predicted_avg_daily']} units/day)."
        )
    return "\n".join(lines)


def generate_summary(summary: dict, category: str, store: str,
                     shap_summary: dict = None) -> str:
    """Generate a plain-English summary of the forecast."""
    context = build_forecast_context(summary, category, store)

    if shap_summary:
        shap_lines = ["\nKey drivers per item:"]
        for item_id, drivers in shap_summary.items():
            driver_str = ", ".join(
                f"{feat} ({'+' if val > 0 else ''}{val})"
                for feat, val in drivers.items()
            )
            shap_lines.append(f"- {item_id}: {driver_str}")
        context += "\n" + "\n".join(shap_lines)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    "Write a 3-4 sentence summary of this forecast for an operations manager. "
                    "Highlight the most important trend and the key drivers behind it. "
                    "End with one concrete action they should take."
                )
            }
        ]
    )
    return response.content[0].text


TOOLS = [
    {
        "name": "get_forecast_summary",
        "description": (
            "Get predicted demand for all items. Shows predicted units/day, "
            "comparison vs prior period, and % change. Use for questions "
            "about demand levels or trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"}
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "get_shap_drivers",
        "description": (
            "Get the top 3 demand drivers for a specific item or all items. "
            "Each driver shows the feature name and its direction of impact. "
            "Use this when asked about WHY demand is changing, what is causing "
            "a trend, or what factors matter most."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "item_id": {
                    "type": "string",
                    "description": "optional, omit to get all items"
                }
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "get_snap_lift",
        "description": (
            "Get the estimated impact of SNAP benefit days on demand, "
            "controlling for other factors. Use when asked about SNAP, "
            "food assistance days, or benefit program effects."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"}
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "get_model_metrics",
        "description": (
            "Get model accuracy (MAE) and the most important features. "
            "Use when asked about model reliability, accuracy, or how "
            "confident we should be in the forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"}
            },
            "required": ["run_id"]
        }
    },
]

COPILOT_SYSTEM = """You are a demand forecasting analyst for a retail supply chain operation.
You have access to tools that query a forecast database.

## About this forecasting system

Model: LightGBM (gradient boosting decision tree)
- Trained on M5 Walmart retail dataset (3,049 SKUs, 5 years of daily sales)
- Features: recent sales trends (7/14/28-day rolling means), lag values,
  price, calendar effects, SNAP indicators
- Evaluation metric: MAE (Mean Absolute Error)

Why LightGBM: Chosen for interpretability, training speed, and strong
performance on tabular retail data. M5 competition top solutions consistently
used gradient boosting over deep learning for this data type.

What this system can and cannot do:
- Can predict general demand trends accurately
- Cannot predict individual spike events (e.g. unplanned promotions)
- Cannot answer counterfactual questions ("what if we run a promotion",
  "what if price drops") — the model has no intervention data
- Can only access the current forecast run, not compare across runs

## Tool usage rules

- Questions about demand levels or trends → get_forecast_summary
- Questions about WHY demand is changing → get_shap_drivers
- Questions about SNAP or food assistance impact → get_snap_lift
- Questions about model accuracy or reliability → get_model_metrics
- Questions about model type, methodology, features → answer from background knowledge above
- Questions outside system scope → clearly state the limitation, do not guess

The percentage  change in forecast_summary compares predicted demand for the next N days
vs actual demand in the prior N days. This measures demand trend, not model error.

Always end with one concrete, actionable recommendation.
Keep answers to 3-5 sentences."""


def answer_question(question: str, run_id: str, history: list = None) -> str:
    from src.database import (get_forecast_summary, get_shap_drivers,
                               get_snap_lift, get_model_metrics)

    fn_map = {
        "get_forecast_summary": get_forecast_summary,
        "get_shap_drivers": get_shap_drivers,
        "get_snap_lift": get_snap_lift,
        "get_model_metrics": get_model_metrics,
    }

    system = (
        COPILOT_SYSTEM
        + f"\n\nCurrent forecast run_id: {run_id}\n"
        "Use this run_id for all tool calls. Never ask the user for a run_id."
    )

    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            system=system,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    inputs = {**block.input, "run_id": run_id}
                    result = fn_map[block.name](**inputs)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str)
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            break

    return "I was unable to answer this question."
