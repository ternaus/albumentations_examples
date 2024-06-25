import argparse
import json
import random
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.text_overlay.utils import prepare_metadata


class DocumentDataset(Dataset):
    def __init__(self, data_path: Path, font_path: Path, transform: A.Compose) -> None:
        self.data_path = data_path
        self.tif_file_paths = sorted(self.data_path.rglob("*.tif"))
        self.fraction_range = (0.1, 0.9)
        self.font_path = font_path
        self.transform = transform

    def __len__(self) -> int:
        # Assuming infinite length as we pick randomly each time
        return len(self.tif_file_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tif_file_path = self.tif_file_paths[idx]
        json_file_path = tif_file_path.with_suffix(".json")

        with json_file_path.open() as f:
            labels = json.load(f)

        img = Image.open(tif_file_path)
        page_id = random.choice(range(img.n_frames))

        img.seek(page_id)
        image_array = np.array(img.convert("RGB"))
        label = labels["pages"][page_id]

        fraction = random.uniform(*self.fraction_range)

        metadata = prepare_metadata(
            image_shape=image_array.shape[:2],
            bboxes=label["bbox"],
            texts=label["text"],
            fraction=fraction,
            font_path=str(self.font_path),
        )

        transformed = self.transform(image=image_array, overlay_metadata=metadata)["image"]

        return {"image": transformed, "id": str(tif_file_path.relative_to(self.data_path).stem)}


def main(input_folder: Path, font_path: Path, output_folder: Path) -> None:
    transform = A.Compose(
        [
            A.OverlayElements(p=1),
            A.RandomCrop(p=1, height=1024, width=1024),
            A.PlanckianJitter(p=1),
            A.Affine(p=1),
        ],
    )

    dataset = DocumentDataset(input_folder, font_path, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    output_folder.mkdir(exist_ok=True, parents=True)

    for batch in tqdm(data_loader):
        batch_size = batch["image"].shape[0]

        for batch_id in range(batch_size):
            id_ = batch["id"][batch_id]
            image = batch["image"][batch_id]
            output_image_path = Path(output_folder) / f"{id_}.png"
            image_pil = Image.fromarray(image.numpy())
            image_pil.save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TAR files to extract images and metadata.")
    parser.add_argument("-i", "--input_path", type=Path, required=True, help="Path to images and documentation")
    parser.add_argument("-f", "--font_path", type=Path, required=True, help="Path to font.")
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output folder to save extracted images and metadata.",
        default="output",
    )

    args = parser.parse_args()

    main(args.input_path, args.font_path, args.output_path)
