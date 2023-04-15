
import logging
from typing import Optional, Union

import torch
import pytorch_lightning as pl
from torch import FloatTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from . import schedulers

class BaseModule(pl.LightningModule):
    """
    Base lightning module for all models. No training or validation steps are
    implemented here.

    Attributes
    ----------
    model: torch.nn.Module
        Model module

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
    _get_optimizer()
        Return optimizer
    _get_scheduler(optimizer)
        Return LR scheduler
    """

    def __init__(
            self, model: torch.nn.Module,
            model_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None,
            scheduler_hparams: Optional[dict] = None
        ) -> None:
        """
        Parameters
        ----------
            model: torch.nn.Module
                Model module
            model_hparams: dict, optional
                Model hyperparameters
            optimizer_hparams: dict, optional
                Optimizer hyperparameters
            scheduler_hparams: dict, optional
                LR scheduler hyperparameters
        """
        super().__init__()
        if model_hparams is None:
            model_hparams = {}
        if optimizer_hparams is None:
            optimizer_hparams = {}
        if scheduler_hparams is None:
            scheduler_hparams = {}

        self.save_hyperparameters()

        # init model, transformation, and optimizer
        self.model = model(**model_hparams)

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
        hparams = self.hparams.optimizer_hparams.copy()
        optimizer = hparams.pop('type')
        if optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), **hparams)
        elif optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), **hparams)
        else:
            raise NotImplementedError(
                "optimizer must be 'Adam' or 'AdamW', not {}".format(optimizer))

    def _get_scheduler(self, optimizer: Optimizer) -> Union[_LRScheduler, None]:
        """ Return LR scheduler """
        hparams = self.hparams.scheduler_hparams.copy()
        scheduler = hparams.pop('type')
        if scheduler is None:
            return None
        elif scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', **hparams)
        elif scheduler == 'AttentionScheduler':
            return schedulers.AttentionScheduler(optimizer, **hparams)
        else:
            raise NotImplementedError(
                "scheduler must be 'ReduceLROnPlateau' or 'AttentionScheduler', "
                "not {}".format(scheduler))


class BaseFlowModule(BaseModule):
    """
    Base lightning module for flow-based models. Training by minimizing the
    log-likelihood P(y | context) where G(x) is the model architecture.
    Explicitly assumes:
    - `self.model` has a `log_prob` method that returns the log-likelihood of the
    data given the context.
    - `self.model` has a `sample` method that returns samples from the model.

    Attributes
    ----------
    model: torch.nn.Module
        Model module

    Methods
    -------
    training_step(batch, batch_idx)
        Training step
    validation_step(batch, batch_idx)
        Validation step
    sample(*args, **kargs)
        Sample from model
    log_prob(*args, **kargs)
        Return log-likelihood of data
    """
    def __init__(
            self, model: torch.nn.Module,
            model_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None,
            scheduler_hparams: Optional[dict] = None
        ) -> None:
        """
        Parameters
        ----------
            model: torch.nn.Module
                Model module
            model_hparams: dict, optional
                Model hyperparameters
            transform_hparams: dict, optional
                Transformation hyperparameters
            optimizer_hparams: dict, optional
                Optimizer hyperparameters
        """
        super().__init__(
            model, model_hparams, optimizer_hparams, scheduler_hparams)

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

    def log_prob(self, *args, **kargs):
        return self.model.log_prob(*args, **kargs)
