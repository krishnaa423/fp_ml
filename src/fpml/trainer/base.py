#region modules
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
#endregions

#region variables
#endregions

#region functions
#endregions

#region classes
class BaseTrainer:
    def __init__(
            self, 
            model: pl.LightningModule, 
            datamodule: pl.LightningDataModule,
            num_nodes: int = 1,
            strategy: str = 'ddp',
            num_epochs: int = 5,
            save_dir: str = './',
            **kwargs
        ):
        self.model: pl.LightningDataModule = model
        self.datamodule: pl.LightningDataModule = datamodule
        self.num_epochs: int = num_epochs
        self.num_nodes: int = num_nodes
        self.strategy: str = strategy
        self.save_dir: str = save_dir
        self.trainer: pl.Trainer = pl.Trainer(
            logger=TensorBoardLogger(save_dir=self.save_dir),
            max_epochs=self.num_epochs,
            accelerator='auto',
            devices='auto',
            num_nodes=self.num_nodes,
            strategy=self.strategy
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)

    def test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def predict(self):
        self.trainer.predict(model=self.model, datamodule=self.datamodule)
#endregions
