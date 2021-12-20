# -*- coding: utf-8 -*-

"""
@File: MyDataset.py
@Author: Chance (Qian Zhen)
@Description: Dataset definition
@Date: 2021/12/06
"""
import os
import glob
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from utils import read_img, fast_read_imgs


class StreetViewDataset(Dataset):
    def __init__(
            self,
            imgs,
            labels,
            img_paths=False,
            transform=None,
    ):
        self.imgs = imgs
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.as_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        if self.img_paths:
            img = read_img(self.imgs[idx])
        else:
            img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            augments = self.transform(image=img)
            return self.as_tensor(augments['image']), label
        else:
            return self.as_tensor(img), label

    def __len__(self):
        if self.imgs is not None:
            return len(self.imgs)
        return len(self.img_paths)


def get_imgs_labels(folder, mode):
    path = os.path.join(folder, "Tier*/%s" % mode)
    postive_img_paths = glob.glob(os.path.join(path, "positive/*.png"))
    negative_img_paths = glob.glob(os.path.join(path, "negative/*.png"))
    img_paths = postive_img_paths + negative_img_paths
    labels = [1] * len(postive_img_paths) + [0] * len(negative_img_paths)
    return img_paths, labels


def get_confusing_imgs_labels(folder):
    img_paths = glob.glob(os.path.join(folder, "SpecialNegtiveCase/*/*.png"))
    labels = [0] * len(img_paths)
    return img_paths, labels


def load_train_data(cfg, train_transform, val_transform, add_confusing_data=False):
    train_img_paths, train_labels = get_imgs_labels(cfg.dataset_path, "train")
    val_img_paths, val_labels = get_imgs_labels(cfg.dataset_path, "val")

    if add_confusing_data:
        confusing_img_paths, confusing_labels = get_confusing_imgs_labels(cfg.dataset_path)
        train_img_paths += confusing_img_paths
        train_labels += confusing_labels

    train_imgs = fast_read_imgs(train_img_paths)
    val_imgs = fast_read_imgs(val_img_paths)

    train_ds = StreetViewDataset(train_imgs, train_labels, transform=train_transform)
    val_ds = StreetViewDataset(val_imgs, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    print("Lenght of training dataset: %d" % len(train_ds))
    print("Lenght of validation dataset: %d" % len(val_ds))

    return train_loader, val_loader


def load_test_data(cfg, test_transform):
    test_img_paths, test_labels = get_imgs_labels(cfg.dataset_path, "test")
    test_imgs = fast_read_imgs(test_img_paths)
    test_ds = StreetViewDataset(test_imgs, test_labels, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.predict_batch_size, shuffle=False)

    print("Length of test dataset: %d" % len(test_ds))

    return test_loader


def load_geoscene_train_data(cfg, train_transform, val_transform, k=10):
    data_path = "../CNN_pretrain/labeling.txt"
    df = pd.read_csv(data_path)
    df["geo_scene"] = np.argmax(df[['建筑', '空旷', '其他']].values, axis=1)

    img_paths = np.array(
        [os.path.join("../CNN_pretrain", path).replace("标注", "labeling").replace("\\", "/") for path in df["path"].tolist()])

    imgs = fast_read_imgs(img_paths)
    labels = df[["geo_scene", "声屏障"]].values

    train_idx = [i for i in range(len(img_paths)) if i % k != 0]
    val_idx = [i for i in range(len(img_paths)) if i % k == 0]

    train_ds = StreetViewDataset(imgs[train_idx], labels[train_idx], transform=train_transform)
    val_ds = StreetViewDataset(imgs[val_idx], labels[val_idx], transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    print("Length of training geoscene dataset: %d" % len(train_ds))
    print("Length of validation geoscene dataset: %d" % len(val_ds))

    return train_loader, val_loader

def load_predict_data(cfg, img_paths, test_transform):
    ds = StreetViewDataset(img_paths, img_paths, img_paths=True, transform=test_transform)
    loader = DataLoader(ds, batch_size=cfg.predict_batch_size, num_workers=4, shuffle=False)

    print("Length of test dataset: %d" % len(ds))
    return loader


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    from utils import Configuration
    import albumentations as A
    from utils import cfg, train_transform, val_transform, test_transform

    # load_train_data(cfg, train_transform, val_transform, add_confusing_data=True)
    # load_test_data(cfg, test_transform)
    load_geoscene_train_data(cfg, train_transform, val_transform)