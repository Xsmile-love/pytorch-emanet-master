import os
import json
from PIL import Image
from PIL import ImageFile
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """Self-defined dataset"""

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)
        # os.remove('./data/mini-imagenet/images/n0211673800000549.jpg')
        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        # labels_dict = {}
        # for i in range(1000):
        #     labels_dict[self.label_dict[str(i)][0]] = i
        # with open("classes_label.json", "w", encoding='utf-8') as f:
        #     # indent Super nice, formatted to save the dictionary, default is None, less than 0 for zero spaces
        #     f.write(json.dumps(labels_dict, indent=4))

        # self.classes_labels = json.load(open(path, "r"))
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB is the color picture, L is the grayscale picture
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # The official implementation of default_collate can be found in
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



