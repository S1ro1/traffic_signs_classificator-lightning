from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import torch
from torchvision.io import read_image 

class TrafficSignsDataset(Dataset):
  def __init__(self, annotations, train, transform=None):
    self.labels = pd.read_csv(annotations)
    self.train = train
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    path = self.labels.iloc[idx]["path"]
    label = self.labels.iloc[idx]["label"]
    
    img = read_image(path)
    
    if self.transform:
      img = self.transform(img)

    return img, label

def get_loader(dataset, annotations, batch_size, weight_sample, shuffle):
  sampler = None
  annotations = pd.read_csv(annotations)
  assert(shuffle != weight_sample)

  if weight_sample:
    weights = []

    num_classes = annotations["label"].nunique()

    sample_weights = [0] * len(dataset)

    for idx in range(num_classes):
      weights.append(1/len(annotations[annotations["label"] == idx]))

    for idx, (data, label) in enumerate(dataset):
      sample_weights[idx] = weights[label]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
  return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
