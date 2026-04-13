import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

PROCESSED_PATH = Path("data/processed.parquet")

def load_processed() -> pd.DataFrame:
    """Load preprocessed parquet file."""
    return pd.read_parquet(PROCESSED_PATH)

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
    """Load from preprocessed parquet and filter to category/store/items."""
    df = load_processed()

    mask = (df["cat_id"] == category) & (df["store_id"] == store)
    df = df[mask].copy()

    # Limit to max_items
    items = df["item_id"].unique()[:max_items]
    df = df[df["item_id"].isin(items)]

    return df.sort_values(["item_id", "date"]).reset_index(drop=True)


def get_available_categories() -> list:
    df = load_processed()
    return sorted(df["cat_id"].unique().tolist())


def get_available_stores() -> list:
    df = load_processed()
    return sorted(df["store_id"].unique().tolist())

def get_available_categories(sales: pd.DataFrame = None) -> list:
    if sales is None:
        sales, _, _ = load_data()
    return sorted(sales["cat_id"].unique().tolist())

def get_available_stores(sales: pd.DataFrame = None) -> list:
    if sales is None:
        sales, _, _ = load_data()
    return sorted(sales["store_id"].unique().tolist())