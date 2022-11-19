from pathlib import Path
import json
from typing import List, Dict, Tuple


DATA_FOLDER = Path("./data")


def get_all_imagenet_files(folder: Path) -> List[Path]:
    files = [file for file in folder.glob("n*.JPEG") if file.is_file()]
    return files


def rename_class_file(
    path: Path, imagenet_map_classes: Dict[str, Tuple[int, str]]
) -> Path:
    img_ix = path.name.split("_")[0]
    c_ix, c_name = imagenet_map_classes[img_ix]
    return path.rename(path.with_name(f"{c_ix}-{c_name}.JPEG"))


files = get_all_imagenet_files(DATA_FOLDER / "image")
print(files[:2])

with open("imagenet_map_classes.json", "r") as f:
    imagenet_map_classes = json.load(f)
    for file in files:
        rename_class_file(file, imagenet_map_classes=imagenet_map_classes)
