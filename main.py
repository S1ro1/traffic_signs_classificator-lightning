import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils import TrafficSignsDataset, get_loader

class CNN(pl.LightningModule):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 3)
        self.p1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.p2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*8*8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _common_step(self, x, batch_idx):

        x = self.p1(F.relu(self.conv1(x)))
        x = self.p2(F.relu(self.conv2(x)))

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)
    
    def training_step(self, batch, batch_idx):
        logger = self.logger.experiment

        x, y = batch
        out = self._common_step(x, batch_idx)
        train_loss = nn.CrossEntropyLoss(out, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 3e-4)
        return optimizer




