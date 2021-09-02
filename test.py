import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Libary for dataset loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.chunk_sampler import ChunkSampler
from utils.mask_generators import MaskGenerators


plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_TRAIN = 50000
NUM_VAL = 5000
NOISE_DIM = 96
BATCH_SIZE = 128

DATA_DIR = r"C:\Users\shabr\OneDrive\Documents\Data\Tugas Akhir\Dataset\img"


def load_data(path: str = ""):
    train_data = datasets.ImageFolder(path, transform=ToTensor())

    return train_data


def show_images(images):
    pass


def main():
    datasets = load_data(path=DATA_DIR)
    loader_train = DataLoader(
        datasets, batch_size=BATCH_SIZE, sampler=ChunkSampler(NUM_TRAIN, 0)
    )

    images, labels = next(iter(loader_train))

    print(labels)

    mask_generator = MaskGenerators(
        num=2,
        height=images[0].shape[1],
        width=images[0].shape[2],
        channels=images[0].shape[0],
    )
    con_mask = mask_generator.continuous_mask(
        max_angle=30, max_length=images[0].shape[2], maxBrushWidth=1
    )
    discon_mask = mask_generator.discontinuous_mask(low=1, high=1)
    # plt.imshow(np.transpose(mask, (1, 2, 0)), interpolation="nearest")
    # plt.show()

    plt.imshow(discon_mask)
    plt.show()


if __name__ == "__main__":
    main()
