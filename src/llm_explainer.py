import anthropic

client = anthropic.Anthropic()  # 默认读 ANTHROPIC_API_KEY 环境变量

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
            f"(from {stats['actual_avg_daily']} → {stats['predicted_avg_daily']} units/day). "
            f"Peak demand expected on {stats['peak_day']} ({stats['peak_predicted']} units)."
        )
    return "\n".join(lines)


def generate_summary(summary: dict, category: str, store: str,
                     shap_summary: dict = None) -> str:
    """Generate a plain-English summary of the forecast."""
    context = build_forecast_context(summary, category, store)

    # Add SHAP context if available
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


def answer_question(question: str, summary: dict,
                    category: str, store: str,
                    history: list = None) -> str:
    """
    Answer a natural language question about the forecast.
    history: list of {"role": ..., "content": ...} dicts for multi-turn support.
    """
    context = build_forecast_context(summary, category, store)

    messages = []

    # Inject forecast context as first user message
    messages.append({
        "role": "user",
        "content": f"Here is the current demand forecast data:\n{context}"
    })
    messages.append({
        "role": "assistant",
        "content": "Got it. I have the forecast data. What would you like to know?"
    })

    # Add conversation history if any
    if history:
        messages.extend(history)

    # Add current question
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    return response.content[0].text