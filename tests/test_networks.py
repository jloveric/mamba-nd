import pytest
from mamba_nd.networks import MambaNDClassificationNet
from mamba_nd.datamodules import MambaDataModule
from omegaconf import DictConfig, OmegaConf
import torch

def test_mamba_network():


    datamodule = MambaDataModule(root_dir="./data")
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()

    cfg = DictConfig(content={
        "dim":5,
        "depth":4,
        "dt_rank" : "auto",
        "d_state": 16,
        "expand_factor": 2,  
        "d_conv": 4,
        "ndimensions": 2,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init": "random",
        "dt_scale": 1.0,
        "dt_init_floor" : 1e-4,
        "bias":False,
        "conv_bias": True,
        "pscan": True,
        "num_classes":10
    })

    x = torch.rand(3,28,28,1)

    network = MambaNDClassificationNet(cfg=cfg)
    y = network(x)

    assert y.shape == torch.Size([x.shape[0], x.shape[1], x.shape[2],cfg.dim])

    print('ans', y)