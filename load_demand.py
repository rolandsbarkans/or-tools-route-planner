from pathlib import Path
from typing import List
import pandas as pd

valid_days: List[str] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
]


def _normalize_day(day_name: str) -> str:
    if not isinstance(day_name, str):
        raise ValueError("Day name must be a string")
    normalized = day_name.strip().lower()
    if normalized not in valid_days:
        raise ValueError(f"Invalid day '{day_name}'. Expected one of {', '.join(valid_days)}")
    return normalized


def load_generated_store_data(
    day_name: str,
    data_dir: Path = Path("data"),
    master_store_path: Path = Path("final_store_data_geocoded.csv"),
) -> pd.DataFrame:
    """Load generated demand data for the specified day and merge with master store info."""
    day = _normalize_day(day_name)

    day_file = data_dir / "demand" / f"{day}_demand.csv"
    if not day_file.exists():
        raise FileNotFoundError(f"Demand file not found for {day_name}: {day_file}")

    if not master_store_path.exists():
        raise FileNotFoundError(f"Master store file not found: {master_store_path}")

    demand_df = pd.read_csv(day_file)
    # Basic validation
    required_demand_cols = {"store_id", "total_v"}
    missing_demand = required_demand_cols - set(demand_df.columns)
    if missing_demand:
        raise ValueError(
            f"Demand file {day_file} is missing required columns: {sorted(missing_demand)}"
        )

    demand_df = demand_df.copy()
    demand_df["store_id"] = demand_df["store_id"].astype(str)
    demand_df["total_v"] = pd.to_numeric(demand_df["total_v"], errors="coerce").fillna(0)
    demand_df = demand_df[demand_df["total_v"] > 0]

    master_df = pd.read_csv(master_store_path)
    required_master_cols = {"store_id", "store_name", "address", "latitude", "longitude"}
    missing_master = required_master_cols - set(master_df.columns)
    if missing_master:
        raise ValueError(
            f"Master store file {master_store_path} is missing required columns: {sorted(missing_master)}"
        )

    master_df = master_df.copy()
    master_df["store_id"] = master_df["store_id"].astype(str)

    merged = demand_df.merge(master_df, on="store_id", how="inner")

    # Use store name from master to ensure consistency, fallback to demand file name if missing
    if "store_name_x" in merged.columns and "store_name_y" in merged.columns:
        merged["store_name"] = merged["store_name_y"].fillna(merged["store_name_x"])
    elif "store_name_y" in merged.columns:
        merged["store_name"] = merged["store_name_y"]
    elif "store_name_x" in merged.columns:
        merged["store_name"] = merged["store_name_x"]
    else:
        merged["store_name"] = merged.get("store_name", merged["store_id"])

    # Ensure final columns exist and are ordered
    final_columns = [
        "store_id",
        "store_name",
        "address",
        "latitude",
        "longitude",
        "total_v",
    ]

    # When demand file already has address/coordinates, prefer master data for accuracy
    merged["address"] = merged["address_y"].fillna(merged.get("address_x")) if "address_y" in merged.columns else merged.get("address", "")
    merged["latitude"] = pd.to_numeric(
        merged.get("latitude_y", merged.get("latitude")), errors="coerce"
    )
    merged["longitude"] = pd.to_numeric(
        merged.get("longitude_y", merged.get("longitude")), errors="coerce"
    )

    cleaned = merged[final_columns].copy()

    # Final validation for nulls
    if cleaned[["latitude", "longitude", "total_v"]].isnull().any().any():
        raise ValueError("Merged dataset contains missing latitude, longitude, or total_v values")

    return cleaned.reset_index(drop=True)
