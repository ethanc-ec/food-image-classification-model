import logging
from pathlib import Path

import polars as pl

DATA_PATH = Path.cwd()

if "src" in str(DATA_PATH):
    DATA_PATH = DATA_PATH.parent / "data"
else:
    DATA_PATH = DATA_PATH / "data"

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
        
        splits = {'train': 'data/train-*.parquet', 'validation': 'data/validation-*.parquet'}
        pl.read_parquet('hf://datasets/ethz/food101/' + splits['train']).write_parquet(DATA_PATH / "food101-train.parquet")
        
        logger.info("Completed: food101-validation dataset")

    else:
        logger.info("food101-train dataset already exists")
        
    if not Path(DATA_PATH / "food101-validation.parquet").exists():
        logger.info("Downloading: food101-validation dataset")
        
        splits = {'train': 'data/train-*.parquet', 'validation': 'data/validation-*.parquet'}
        pl.read_parquet('hf://datasets/ethz/food101/' + splits['validation']).write_parquet(DATA_PATH / "food101-validation.parquet")
        
        logger.info("Completed: food101-validation dataset")

    else:
        logger.info("food101-validation dataset already exists")

    if not Path(DATA_PATH / "food102-train.parquet").exists():
        logger.info("Downloading: food102-train dataset")
    
        splits = {'train': 'data/train-*-of-*.parquet', 'test': 'data/test-*-of-*.parquet'}
        pl.read_parquet('hf://datasets/juliensimon/food102/' + splits['train']).write_parquet(DATA_PATH / "food102-train.parquet")
        
        logger.info("Completed: food102-train dataset")
    
    else:
        logger.info("food102-train dataset already exists")
        
    if not Path(DATA_PATH / "food102-test.parquet").exists():
        logger.info("Downloading: food102-test dataset")

        splits = {'train': 'data/train-*-of-*.parquet', 'test': 'data/test-*-of-*.parquet'}
        pl.read_parquet('hf://datasets/juliensimon/food102/' + splits['test']).write_parquet(DATA_PATH / "food102-test.parquet")
        
        logger.info("Completed: food102-test dataset")
    
    else:
        logger.info("food102-test dataset already exists")
        
    logger.info("Completed: All downloads/checks")
