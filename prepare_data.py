from src.data_loader import load_data, melt_sales
import pandas as pd

print("Loading raw data...")
sales, calendar, prices = load_data()

# 聚合价格，一个 (store_id, item_id) 一行
prices_agg = (prices.groupby(["store_id", "item_id"])["sell_price"]
                    .mean().reset_index())

results = []
for category in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
    for store in ["CA_1", "CA_2", "TX_1"]:
        mask = (sales["cat_id"] == category) & (sales["store_id"] == store)
        subset = sales[mask].head(20)
        if subset.empty:
            continue

        df = melt_sales(subset)
        df = df.merge(calendar[["d", "date", "wday", "month", "year",
                                 "event_name_1", "event_type_1",
                                 "snap_CA", "snap_TX", "snap_WI"]],
                      on="d", how="left")
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(prices_agg, on=["store_id", "item_id"], how="left")
        df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
        results.append(df)
        print(f"Done: {category} / {store} — {df.shape}")

final = pd.concat(results, ignore_index=True)
final.to_parquet("data/processed.parquet", index=False)
print(f"\nSaved data/processed.parquet — {final.shape}")
print(f"File size: {final.memory_usage(deep=True).sum() / 1e6:.1f} MB (in memory)")