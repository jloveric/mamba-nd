from mamba_nd.networks import MambaNDClassificationNet
from mamba_nd.datamodules import MambaDataModule
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

import hydra
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

@hydra.main(
    config_path="../configs", config_name="config", version_base="1.3"
)
def run(cfg: DictConfig):
    
    datamodule = MambaDataModule()
    model = MambaNDClassificationNet(cfg=cfg)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(monitor="train_loss", patience=20)
    trainer = Trainer(
        callbacks=[early_stopping, lr_monitor],
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        gradient_clip_val=cfg.gradient_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    logger.info("testing")

    result = trainer.test(model, datamodule=datamodule)
    logger.info(f"result {result}")
    logger.info("finished testing")
    logger.info(
        f"best check_point {trainer.checkpoint_callback.best_model_path}"
    )
    logger.info(f"loss {result[0]['test_loss']}")
    return result[0]["test_loss"]


if __name__ == "__main__":
    run()