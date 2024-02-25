import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from lightning.pytorch import LightningDataModule
import torchvision
import logging
from torch import Tensor
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class MambaDataModule(LightningDataModule):
    def __init__(
        self,
        targets: int = 1,
        batch_size: int = 32,
        num_workers: int = 10,
        shuffle: bool = True,
        pin_memory: bool = False,
        pre_process_workers: int = 10,
        root_dir: str = ".",
        as_index: bool = False,
    ):
        """
        Data module for this type of transformer
        :param max_size: Truncate the loaded data to "max_size", when -1 is
        used the entire text is used.
        """
        super().__init__()

        self._targets = targets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._pre_process_workers = pre_process_workers
        self._root_dir = root_dir
        self._as_index = as_index

    def collate_fn(self, batch) -> tuple[Tensor, Tensor, list[int]]:

        images = torch.stack([torch.permute(image,[1,2,0]) for image, classification in batch])
        classification = torch.tensor([classification for image, classification in batch])

        return images, classification

    def setup(self, stage: Optional[str] = None):

        self._train_dataset = torchvision.datasets.MNIST(
            root=self._root_dir, train=True, download=True,transform=transforms.ToTensor()
        )

        # Ok, I need to redo both of these
        self._val_dataset = torchvision.datasets.MNIST(
            root=self._root_dir, train=False, download=True
        )

        self._test_dataset = torchvision.datasets.MNIST(
            root=self._root_dir, train=False, download=True
        )

        logger.info(f"Training dataset has {len(self.train_dataset)} samples.")
        logger.info(f"Validation dataset has {len(self.val_dataset)} samples.")
        logger.info(f"Test dataset has {len(self.test_dataset)} samples.")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
