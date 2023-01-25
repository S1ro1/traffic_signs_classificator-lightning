import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils import TrafficSignsDataset, get_loader
import os


class CNN(pl.LightningModule):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=3)
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
        x = x.float()

        out = self._common_step(x, batch_idx)
        loss = F.cross_entropy(out, y)

        n_correct = (out.argmax(1) == y).sum().item()
        accuracy = n_correct/x.shape[0]

        self.log("train accuracy", accuracy, prog_bar=True)
        self.log("train loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        logger = self.logger.experiment

        x, y = batch
        x = x.float()

        out = self._common_step(x, batch_idx)
        loss = F.cross_entropy(out, y)

        n_correct = (out.argmax(1) == y).sum().item()
        accuracy = n_correct/x.shape[0]

        self.log("test accuracy", accuracy)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 3e-4)
        return optimizer


if __name__ == "__main__":
    logger = pl_loggers.WandbLogger(
        project="Traffic-signs-lightning", log_model="all")

    train_dataset = TrafficSignsDataset(
        "resized_data/train-annotations.csv", True)
    test_dataset = TrafficSignsDataset(
        "resized_data/test-annotations.csv", False)

    train_dataloader = get_loader(
        train_dataset, "resized_data/train-annotations.csv", 64, False, shuffle=True)
    test_dataloader = get_loader(
        test_dataset, "resized_data/test-annotations.csv", 64, False, shuffle=True)

    model = CNN(num_classes=43)
    checkpoint = "Traffic-signs-lightning/e9f2ff3t/checkpoints/epoch=6-step=4291.ckpt"

    trainer = pl.Trainer(max_epochs=30, accelerator='gpu',
                         devices=1, logger=logger)

    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=checkpoint)
    trainer.test(model, test_dataloader)

