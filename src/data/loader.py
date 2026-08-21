"""Data Loader for Customer Support Tickets."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


class DataLoader:
    """Class to handle reading raw and processed ticket data."""

    def __init__(self, raw_path: Optional[Union[str, Path]] = None):
        self.raw_path = Path(raw_path) if raw_path else None

    def load_raw_data(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load raw CSV data into a pandas DataFrame."""
        path = Path(filepath) if filepath else self.raw_path
        if not path or not path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {path}")
        return pd.read_csv(path)

    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Save processed DataFrame to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
