"""Processes images from parquet files and converts them to JPG format.

It uses a ThreadPoolExecutor to parallelize the processing of images.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import repeat
from json import dump
from pathlib import Path

import polars as pl
from PIL import Image

DATA_PATH = Path.cwd().parent / "data" if Path.cwd().name == "src" else Path.cwd() / "data"

def process_base(row: tuple, target_dataset: str) -> tuple:
    """Process a single image row and saves it as a JPG file.

    Args:
        row (tuple): A tuple containing the image bytes, filename, and label.
        target_dataset (str): The name of the target dataset.

    Returns:
        tuple: A tuple containing the filename and label.

    """
    image, filename, label = row
    Image.open(BytesIO(image)).save(f"{DATA_PATH}/base_jpg/{target_dataset}/{filename}")
    return filename, label

if __name__ == "__main__":
    for dataset in ["food101-train", "food101-validation"]:
        print(f"Processing {dataset}")

        data = pl.scan_parquet(
            DATA_PATH / f"{dataset}.parquet",
        ).select(
            pl.col("image"),
            pl.col("label"),
        ).unnest("image").collect()

        print(f"Read {data.select(pl.len()).item()} images")

        Path(f"{DATA_PATH}/base_jpg/{dataset}").mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            results = dict(executor.map(process_base, data.iter_rows(), repeat(dataset)))

        print(f"Processed {len(results)} images")

        with Path(DATA_PATH / f"{dataset}-mappings.json").open("w") as f:
            dump(results, f, indent=4)

        print(f"Saved mappings to {dataset}-mappings.json")
