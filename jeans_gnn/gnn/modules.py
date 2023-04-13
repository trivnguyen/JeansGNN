
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

import torch
import pytorch_lightning as pl
from torch import FloatTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from . import schedulers

class BaseModule(pl.LightningModule):
    """ Base lightning module with models and preprocessing transforms

    Attributes
    ----------
    model: torch.nn.Module
        Model module
    transform: torch.nn.Module
        Transformation module

    Methods
    -------
    forward(x, *args, **kargs)
        Forward pass
    configure_optimizers()
        Initialize optimizer and LR scheduler
    training_step(batch, batch_idx)
        Training step
    validation_step(batch, batch_idx)
        Validation step
    test_step(batch, batch_idx)
        Test step
    _get_optimizer()
        Return optimizer
    _get_scheduler(optimizer)
        Return LR scheduler
    """

    def __init__(
            self, model: torch.nn.Module, transform: torch.nn.Module,
            model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        """
        Parameters
        ----------
            model: torch.nn.Module
                Model module
            transform: torch.nn.Module
                Transformation module
            model_hparams: Optional(dict).
                Model hyperparameters. Default: None
            transform_hparams: Optional(dict).
                Transformation hyperparameters. Default: None
            optimizer_hparams: Optional(dict).
                Optimizer hyperparameters. Default: None

        """
        super(BaseModule, self).__init__()
        self.save_hyperparameters()

        if model_hparams is None:
            model_hparams = {}
        if transform_hparams is None:
            transform_hparams = {}
        if optimizer_hparams is None:
            optimizer_hparams = {"optimizer": {}, "scheduler": {}}

        # print out hyperparameters
        logger.info("Hyperparameters:")
        for hparams in self.hparams:
            logger.info(f"{hparams}: {self.hparams[hparams]}")

        # init model, transformation, and optimizer
        self.model = model(**model_hparams)
        self.transform = transform(**transform_hparams)

    def forward(self, x, *args, **kargs):
        """ Forward pass """
        return self.model(x, *args, **kargs)

    def configure_optimizers(self) -> Union[Optimizer, dict]:
        """ Initialize optimizer and LR scheduler """
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def _get_optimizer(self) -> Optimizer:
        hparams = self.hparams.optimizer_hparams['optimizer']
        optimizer = hparams.pop('optimizer')
        if optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), **hparams)
        elif optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), **hparams)
        else:
            raise NotImplementedError(
                "optimizer must be 'Adam' or 'AdamW', not {}".format(optimizer))

    def _get_scheduler(self, optimizer: Optimizer) -> Union[_LRScheduler, None]:
        """ Return LR scheduler """
        hparams = self.hparams.optimizer_hparams['scheduler']
        scheduler = hparams.pop('scheduler')
        if scheduler is None:
            return None
        elif scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', **hparams)
        elif scheduler == 'AttentionScheduler':
            print(hparams)
            return schedulers.AttentionScheduler(optimizer, **hparams)
        else:
            raise NotImplementedError(
                "optimizer must be 'ReduceLROnPlateau or AttentionScheduler'"\
                ", not {}".format(scheduler))


class MAFModule(BaseModule):
    """
    Masked autoregressive flow (MAF) data module inherited from BaseModule.
    Training by minimizing the log-likelihood P(y | context) where G(x) is the
    model architecture.
    Attributes
    ----------
    model: torch.nn.Module
        Model module
    transform: torch.nn.Module
        Transformation module

    Methods
    -------
    training_step(batch, batch_idx)
        Training step
    validation_step(batch, batch_idx)
        Validation step
    test_step(batch, batch_idx)
        Test step
    """
    def __init__(
            self, model: torch.nn.Module, transform: torch.nn.Module,
            model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        """
        Parameters
        ----------
            model: torch.nn.Module
                Model module
            transform: torch.nn.Module
                Transformation module
            model_hparams: Optional(dict).
                Model hyperparameters. Default: None
            transform_hparams: Optional(dict).
                Transformation hyperparameters. Default: None
            optimizer_hparams: Optional(dict).
                Optimizer hyperparameters. Default: None

        """
        super(MAFModule, self).__init__(model, transform, model_hparams,
                                        transform_hparams, optimizer_hparams)

    def training_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch)

        # apply forward and return log-likelihood and loss
        log_prob = self.model.log_prob(batch)
        loss = -log_prob.mean()

        # log loss and return
        self.log('train_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch)

        # apply forward and return log-likelihood and loss
        log_prob = self.model.log_prob(batch)
        loss = -log_prob.mean()

        # log loss and return
        self.log('val_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss

    def sample(self, *args, **kargs):
        return self.model.sample(*args, **kargs)
