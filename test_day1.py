from src.data_loader import build_dataset
from src.forecaster import train_forecast, summarise_forecast

print("Building dataset...")
df = build_dataset(category="FOODS", store="CA_1", max_items=5)
print(f"Dataset shape: {df.shape}")
print(df[["item_id", "date", "sales", "sell_price"]].head(10))

print("\nTraining model...")
forecast_df, mae, importance = train_forecast(df, horizon=14)
print(f"MAE: {mae:.3f}")
print(f"\nTop features:\n{importance.head(5)}")

summary = summarise_forecast(forecast_df)
print(f"\nForecast summary (first item):")
first_item = list(summary.keys())[0]
print(summary[first_item])


import os
from src.llm_explainer import generate_summary, answer_question

assert os.environ.get("ANTHROPIC_API_KEY"), "请先设置 ANTHROPIC_API_KEY 环境变量"

print("\n--- LLM Summary ---")
narrative = generate_summary(summary, category="FOODS", store="CA_1")
print(narrative)

print("\n--- Q&A Test ---")
answer = answer_question(
    question="Which item should I prioritize restocking next week?",
    summary=summary,
    category="FOODS",
    store="CA_1"
)
print(answer)

