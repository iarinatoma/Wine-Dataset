import pandas as pd
import numpy as np


def load_and_clean(path: str, save_path: str = None) -> pd.DataFrame:
    """
    Loads the raw Wine Dataset and applies all cleaning steps.

    Steps:
    - Extracts numeric Price from string (e.g. "£9.99 per bottle" -> 9.99)
    - Applies log-transform to Price
    - Drops rows with missing Price or Description
    - Drops high-missing columns
    - Cleans ABV from string (e.g. "ABV 13.50%" -> 13.50)
    - Fills missing categorical values with 'Unknown'
    - Fills missing ABV with median

    Args:
        path: path to raw CSV file
        save_path: optional path to save cleaned CSV (e.g. 'data/processed/WineDataset_clean.csv')

    Returns:
        cleaned DataFrame
    """

    df = pd.read_csv(path)

    # --- Price cleaning ---
    df['Price_clean'] = df['Price'].str.extract(r'£([\d.]+)').astype(float)
    df['Price_log'] = np.log1p(df['Price_clean'])

    # --- Drop rows with missing Price or Description ---
    df = df.dropna(subset=['Price_clean', 'Description'])

    # --- Drop high-missing / irrelevant columns ---
    df = df.drop(columns=[
        'Secondary Grape Varieties', 'Appellation',
        'Price', 'Title', 'Per bottle / case / each',
        'Characteristics', 'Unit'
    ])

    # --- ABV cleaning ---
    df['ABV'] = df['ABV'].str.extract(r'([\d.]+)').astype(float)

    # --- Fill missing values ---
    cat_cols = ['Grape', 'Closure', 'Country', 'Type',
                'Region', 'Style', 'Vintage', 'Capacity']
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    df['ABV'] = df['ABV'].fillna(df['ABV'].median())

    # --- Save if path provided ---
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Cleaned data saved to: {save_path}")

    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Remaining columns: {df.columns.tolist()}")

    return df