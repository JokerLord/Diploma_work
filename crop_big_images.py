import glob
from pathlib import Path
from PIL import Image


FOLDER_PATH = "D:\\Learning\\MSU\\4th\\diploma_files"

IMAGE_SIZE = 5000

NOISE_LEVEL: int = 0
NUM_OF_CLASSES: int = 20
NUM_OF_IMAGES: int = 5


def crop_big_images(input_folder_path: Path, output_folder_path: Path) -> None:
    output_folder_path.mkdir(parents=True, exist_ok=True)
    for class_num in range(NUM_OF_CLASSES):
        for image_num in range(NUM_OF_IMAGES):
            image = Image.open(input_folder_path /
                               Path(f"{class_num}_{image_num}.png"))
            image.crop((0, 0, IMAGE_SIZE,
                        IMAGE_SIZE)).save(output_folder_path /
                                          Path(f"{class_num}_{image_num}.jpg"))


if __name__ == "__main__":
    crop_big_images(Path(FOLDER_PATH) / Path(f"full_test_noise{NOISE_LEVEL}"),
                    Path(FOLDER_PATH /
                    Path(f"full_test_noise{NOISE_LEVEL}_cropped")))
