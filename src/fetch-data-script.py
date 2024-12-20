#!/usr/bin/env python3

"""Fetches and saves the food101 dataset in parquet format."""

import logging
from pathlib import Path

import polars as pl

DATA_PATH = (Path.cwd().parent / "data") if "src" in str(Path.cwd()) else (Path.cwd() / "data")

if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"Using DATA_PATH: {DATA_PATH}")

    if not Path(DATA_PATH / "food101-train.parquet").exists():
        logger.info("Downloading: food101-train dataset")

        splits = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet"}
        pl.read_parquet("hf://datasets/ethz/food101/" + splits["train"]).write_parquet(DATA_PATH / "food101-train.parquet")

        logger.info("Completed: food101-validation dataset")

    else:
        logger.info("food101-train dataset already exists")

    if not Path(DATA_PATH / "food101-validation.parquet").exists():
        logger.info("Downloading: food101-validation dataset")

        splits = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet"}
        pl.read_parquet("hf://datasets/ethz/food101/" + splits["validation"]).write_parquet(DATA_PATH / "food101-validation.parquet")

        logger.info("Completed: food101-validation dataset")

    else:
        logger.info("food101-validation dataset already exists")

    logger.info("Completed: All downloads/checks")
