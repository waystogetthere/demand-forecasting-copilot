import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def load_data():
    """Load and merge M5 datasets."""
    print("Loading sales data...")
    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    return sales, calendar, prices

def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format sales to long-format."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    melted = sales.melt(id_vars=id_cols, value_vars=day_cols,
                        var_name="d", value_name="sales")
    return melted

def build_dataset(category: str = "FOODS", store: str = "CA_1",
                  max_items: int = 10) -> pd.DataFrame:
    sales, calendar, prices = load_data()

    mask = (sales["cat_id"] == category) & (sales["store_id"] == store)
    sales_filtered = sales[mask].head(max_items)

    df = melt_sales(sales_filtered)

    # Merge calendar
    df = df.merge(calendar[["d", "date", "wday", "month", "year",
                             "event_name_1", "event_type_1",
                             "snap_CA", "snap_TX", "snap_WI"]],
                  on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate prices to one row per (store_id, item_id) — use mean price
    prices_agg = (prices.groupby(["store_id", "item_id"])["sell_price"]
                        .mean()
                        .reset_index())
    df = df.merge(prices_agg, on=["store_id", "item_id"], how="left")

    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    return df
def get_available_categories(sales: pd.DataFrame = None) -> list:
    if sales is None:
        sales, _, _ = load_data()
    return sorted(sales["cat_id"].unique().tolist())

def get_available_stores(sales: pd.DataFrame = None) -> list:
    if sales is None:
        sales, _, _ = load_data()
    return sorted(sales["store_id"].unique().tolist())