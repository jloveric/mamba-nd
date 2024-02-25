from mamba_nd.datamodules import MambaDataModule
import torch

def test_mamba_datamodule() :
    datamodule = MambaDataModule(root_dir="./data")
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()

    x,y =next(iter(train_dataloader))

    assert x.shape == torch.Size([32,28,28,1])
    assert y.shape == torch.Size([32])

