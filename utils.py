# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance (Qian Zhen)
@Description: configuration of this study
@Date: 2021/12/06
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import cv2
import concurrent.futures
import albumentations as A

class Configuration:
    def __init__(self, model_name, output_num, batch_size, predict_batch_size,
                 crop_size, train_image_size, val_image_size, lr, weight_decay,
                 dataset_path=None, lr_warmup_epochs=5, epochs=60,
                 device="cuda", save_path=""):
        self.model_name = model_name
        self.output_num = output_num
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.crop_size = crop_size
        self.train_image_size = train_image_size
        self.val_image_size = val_image_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset_path = dataset_path
        self.lr_warmup_epochs = lr_warmup_epochs
        self.epochs = epochs
        self.device = device
        self.save_path = save_path

def read_img(img_path):
    return cv2.imread(img_path)[..., ::-1]

def fast_read_imgs(img_pahts):
    imgs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img in executor.map(read_img, img_pahts):
            imgs.append(img)
    imgs = np.array(imgs)
    return imgs


cfg = Configuration(
    model_name="ResNet101",
    dataset_path="../CNN_pretrain/dataset",
    output_num=2,
    batch_size=80,
    predict_batch_size=384,
    crop_size=400,
    train_image_size=304,
    val_image_size=400,
    lr=0.1,
    weight_decay=1e-4,
    lr_warmup_epochs=3,
    epochs=100,
    save_path="./ckpts"
)

train_transform = A.Compose([
        A.RandomCrop(cfg.crop_size, cfg.crop_size),
        A.Resize(cfg.train_image_size, cfg.train_image_size),
        A.HorizontalFlip(p=0.5)
    ])

val_transform = A.Compose([
    A.Resize(cfg.val_image_size, cfg.val_image_size)
])

test_transform = A.Compose([
    A.Resize(cfg.val_image_size, cfg.val_image_size)
])