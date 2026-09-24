import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# ============================================================
# Config
# ============================================================

DATA_DIR = Path("feature_library")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(data_dir):
    class_to_images = {}

    for class_dir in sorted(data_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        image_paths = []

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue

            image_paths.append(image_path)

        if image_paths:
            class_to_images[class_dir.name] = image_paths

    return class_to_images

def load_image(self, image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        image = transform(image)
    return image

class TripletDataset(Dataset):
    def __init__(
            self,
            class_to_images,
            transform=None,
            samples_per_epoch=10000,
    ):
        self.class_to_images = class_to_images
        self.transform = transform
        self.classes = list(class_to_images.keys())
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch



    def __getitem__(self, index):
        anchor_class = random.choice(self.classes)
        images = self.class_to_images[anchor_class]
        anchor, positive = random.sample(images, 2)
        negative_class = random.choice([
            class_name for class_name in self.classes if class_name != anchor_class
        ])
        negative = random.choice(self.class_to_images[negative_class])

        return (load_image(image_path=anchor, transform=self.transform),
                load_image(image_path=positive, transform=self.transform),
                load_image(image_path=negative, transform=self.transform))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    class_to_images = load_dataset(
        DATA_DIR
    )

    for class_name, image_paths in class_to_images.items():
        print(
            f"{class_name:<20} "
            f"{len(image_paths)} images"
        )

    dataset = TripletDataset(class_to_images, samples_per_epoch=1000)
    print(type(dataset))
    print("Dataset size:", len(dataset))
    for i in range(3):
        anchor, positive, negative = dataset[i]
        print()
        print("Triplet", i)
        print("Anchor:  ", anchor)
        print("Positive:", positive)
        print("Negative:", negative)
