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
from typing import Tensor
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class MambaDataModule(LightningDataModule):
    def __init__(
        self,
        characters_per_feature: int,
        max_features: int,
        targets: int = 1,
        batch_size: int = 32,
        num_workers: int = 10,
        shuffle: bool = True,
        pin_memory: bool = False,
        pre_process_workers: int = 10,
        max_size: int = -1,
        root_dir: str = ".",
        add_channel_dimension: bool = False,
        as_index: bool = False,
    ):
        """
        Data module for this type of transformer
        :param max_size: Truncate the loaded data to "max_size", when -1 is
        used the entire text is used.
        """
        super().__init__()
        self._characters_per_feature = characters_per_feature

        self._max_features = max_features

        self._targets = targets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._pre_process_workers = pre_process_workers
        self._max_size = max_size
        self._root_dir = root_dir
        self._add_channel_dimension = add_channel_dimension
        self._as_index = as_index

    def collate_fn(self, batch) -> tuple[Tensor, Tensor, list[int]]:
        # TODO: this does not make sense to me
        # The max size includes the output

        max_size = max(self._max_size, batch[0][0].shape[0])
        this_size = random.randint(1, max_size - 1)
        final_features = torch.stack([sample[0][:this_size] for sample in batch])

        # grab the first letter of the next token
        final_targets = torch.stack([sample[0][this_size][0] for sample in batch])

        final_indexes = [sample[1] for sample in batch]
        if self._as_index is True:
            return (
                final_features,
                final_targets,
                final_indexes,
            )

        return self.normalize(final_features), final_targets, final_indexes

    def setup(self, stage: Optional[str] = None):

        self._train_dataset = torchvision.datasets.MNIST(
            root=self._root_dir, train=True, download=True
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
