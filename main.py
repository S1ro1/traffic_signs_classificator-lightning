import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.cli import LightningCLI
from utils import TrafficSignsDataset, get_loader
import torchvision.models as models


class CNN(pl.LightningModule):
    def __init__(self, num_classes=43, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(7*7*128, 128, )
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_accuracy(self, out, y):
        n_correct = (out.argmax(1) == y).sum().item()
        accuracy = n_correct/y.shape[0]

        return accuracy

    def _common_step(self, x, batch_idx):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.dropout(self.fc1(x)))

        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        logger = self.logger.experiment

        x, y = batch
        x = x.float()

        out = self._common_step(x, batch_idx)
        loss = F.cross_entropy(out, y)

        accuracy = self._calculate_accuracy(out, y)

        self.log("train accuracy", accuracy, prog_bar=True)
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logger = self.logger.experiment

        x, y = batch
        x = x.float()

        out = self._common_step(x, batch_idx)
        loss = F.cross_entropy(out, y)

        accuracy = self._calculate_accuracy(out, y)

        self.log("val accuracy", accuracy, prog_bar=True)
        self.log("val loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        logger = self.logger.experiment

        x, y = batch
        x = x.float()

        out = self._common_step(x, batch_idx)
        loss = F.cross_entropy(out, y)

        accuracy = self._calculate_accuracy(out, y)

        self.log("test accuracy", accuracy)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 3e-4)
        return optimizer


class TrafficDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = TrafficSignsDataset(
            "resized_data/train-annotations.csv", False)
        self.test_dataset = TrafficSignsDataset(
            "resized_data/test-annotations.csv", False)

    def train_dataloader(self):
        return get_loader(self.train_dataset, "resized_data/train-annotations.csv", self.batch_size, False, shuffle=True, num_workers=0)

    def test_dataloader(self):
        return get_loader(self.test_dataset, "resized_data/test-annotations.csv", self.batch_size, False, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return self.test_dataloader()

if __name__ == "__main__":
    logger = pl_loggers.WandbLogger(
        project="Traffic-signs-l2", log_model="all")

    model = CNN()
    data_module = TrafficDataModule()

    trainer = pl.Trainer(max_epochs=30, accelerator='gpu',
                         devices=1, logger=logger)

    trainer.fit(model=model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)
