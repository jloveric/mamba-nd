from mamba_nd.datamodules import MambaDataModule

def test_mamba_datamodule() :
    a = MambaDataModule()
    a.setup()