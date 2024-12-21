import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class HAM10000(Dataset):
  num_classes = 7

  def __init__(self, txt, transform=None):
    self.img_path = []
    self.targets = []
    self.transform = transform
    with open(txt, 'r') as f:
      for line in f:
        self.img_path.append(line.split()[0])
        self.targets.append(int(line.split()[1]))

  def __len__(self):
    return len(self.img_path)

  def __getitem__(self, index):
    path = os.path.join(self.img_path[index])
    target = self.targets[index]

    with open(path, 'rb') as f:
      image = Image.open(f).convert('RGB')
    if self.transform is not None:
      image = self.transform(image)

    return image, target


class HAM10000_Dataset:

  def __init__(self, batch_size=60, num_workers=4) -> None:
    self.batch_size = batch_size
    self.num_workers = num_workers

    train_txt = '/Volumes/T7/DL/CSA/CSA/datasets/data_txt/ham1000_train.txt'
    test_txt = '/Volumes/T7/DL/CSA/CSA/datasets/data_txt/ham1000_test.txt'

    transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HAM10000(train_txt, transform=transform)
    test_dataset = HAM10000(test_txt, transform=transform)

    self.train_instance = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers)
    self.test_eval = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers)
