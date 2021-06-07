import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import transforms
from PIL import Image
import glob
import os


class CycleganDataset(Dataset):
    def __init__(self, root, transform_aug=None, mode="train"):
        self.root = root
        self.model = mode
        self.transform_aug = transforms.Compose(transform_aug)

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        if self.transform:
            image_A = self.transform_aug(image_A)
            image_B = self.transform_aug(image_B)

        return torch.tensor(image_A), torch.tensor(image_B)
