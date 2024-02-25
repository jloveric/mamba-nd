import pytest
from mamba_nd.networks import MambaNDClassificationNet
from mamba_nd.datamodules import MambaDataModule


def test_mamba_network():
    
    network = MambaNDClassificationNet