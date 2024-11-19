
"""Processes images to remove their background using the rembg library."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from rembg import new_session, remove
from torch.cuda import is_available
from tqdm import tqdm

session = new_session("u2net",  ["CUDAExecutionProvider"] if is_available() else ["CPUExecutionProvider"])
DATA_PATH = Path.cwd().parent / "data" if Path.cwd().name == "src" else Path.cwd() / "data"
REGENERATE = False

def process_image(image_path: Path) -> None:
    """Process an image to remove its background and save the result.

    Args:
        image_path (Path): The path to the image file to be processed.

    """
    sink = Path(str(image_path).replace("base_jpg", "no_bg_png").replace("jpg", "png"))
    if REGENERATE or not sink.exists():
        with Image.open(image_path) as img:
            cleaned = remove(img, session=session, bgcolor=(0, 0, 0, 0), post_process_mask=True)
            cleaned.save(sink, "PNG")

if __name__ == "__main__":
    for dataset in ["food101-train", "food101-validation"]:
        print(f"Fetching {dataset} files")
        all_files = list(Path(f"{DATA_PATH}/base_jpg/{dataset}").glob("*.jpg"))
        print(f"Found {len(all_files)} images")

        Path(f"{DATA_PATH}/base_jpg/{dataset}").mkdir(parents=True, exist_ok=True)
        Path(f"{DATA_PATH}/no_bg_png/{dataset}").mkdir(parents=True, exist_ok=True)

        print(f"Processing {dataset}")
        with tqdm(total=len(all_files), desc=dataset) as pbar, ThreadPoolExecutor() as executor:
                for _ in executor.map(process_image, all_files):
                    pbar.update(1)
