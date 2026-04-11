# Demand Forecasting Copilot

Demand forecasting for supply chain operations. Combines LightGBM time-series forecasting with Claude AI to generate
actionable insights for operations teams — not just numbers, but recommendations.

![Forecast Overview](assets/screenshot.png)

## What it does

- Forecasts daily product demand using LightGBM trained on historical sales,
  pricing, calendar, and promotional signals
- Generates plain-English summaries of forecast trends via Claude AI
- Answers natural language questions about inventory priorities
- Handles edge cases cleanly (near-zero baseline items, seasonal anomalies)

## Dataset

[M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
— Walmart daily sales data across 3,049 products, 10 stores, 1,913 days.
Includes pricing, calendar events, and SNAP indicators.

Data is not included in this repo. Download from Kaggle and place CSV files in `data/`.

## Stack

- **Forecasting:** LightGBM, pandas, scikit-learn
- **LLM:** Anthropic Claude API
- **UI:** Streamlit, Plotly

## Quickstart

```bash
git clone https://github.com/waystogetthere/biopak-demand-copilot
cd biopak-demand-copilot
pip install -r requirements.txt

# Add data files to data/ from Kaggle (see Dataset section above)

export ANTHROPIC_API_KEY='sk-ant-...'
streamlit run app.py
```

## Model notes

The model predicts expected demand trends accurately but underestimates
spike events not captured in the feature set (e.g. unlogged promotions,
localised events). MAE on the 14-day holdout set: ~0.63 units/day across
FOODS category items at a single store.

Future improvements: incorporate external promotion signals, use walk-forward
validation for more robust evaluation, add SHAP-based explanations to the
LLM context.