import torch
import lightning as pl
from model import HU_PageScan
from data import DocDataModule
# from callbacks import MyPrintingCallback, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium") # to make lightning happy


if __name__ == "__main__":

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        min_epochs=1, 
        max_epochs=10000, 
        precision='16-mixed',
        default_root_dir="checkpoints",
        check_val_every_n_epoch=10,
        enable_checkpointing = True
    )

    model = HU_PageScan(4,3)
    # model.load_from_checkpoint("checkpoints/lightning_logs/version_3/checkpoints/epoch=29-step=13650.ckpt")
    
    dm = DocDataModule(
        json_path="SmartDocExtended.json",
        data_dir="SmartDocExtended",
        batch_size=32,
        num_workers=4
    )
    
    trainer.fit(model, dm, ckpt_path="checkpoints/lightning_logs/version_27/checkpoints/epoch=29-step=13650.ckpt")
    # trainer.validate(model, dm)
    # trainer.test(model, dm, ckpt_path="checkpoints/lightning_logs/version_3/checkpoints/epoch=29-step=13650.ckpt")


