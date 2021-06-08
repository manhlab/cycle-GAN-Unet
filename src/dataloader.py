from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


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
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        if self.transform_aug:
            image_A = self.transform_aug(image_A)
            image_B = self.transform_aug(image_B)

        return image_A, image_B
