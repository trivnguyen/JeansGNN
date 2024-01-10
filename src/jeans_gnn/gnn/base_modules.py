
import logging
from typing import Optional, Union

import torch
import pytorch_lightning as pl
from torch import FloatTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from . import schedulers
from . import transforms

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

        # create optimizer
        optimizer_args = self.hparams.optimizer_hparams.copy()
        optimizer_name = optimizer_args.pop('type')
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)
        elif optimizer_name  == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_args)
        else:
            raise NotImplementedError(
                'optimizer not implemented: {}'.format(optimizer_name))

        # create scheduler
        scheduler_args = self.hparams.scheduler_hparams.copy()
        scheduler_name = scheduler_args.pop('type')
        scheduler_interval = scheduler_args.pop('interval')
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_args)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_args)
        elif scheduler_name == "WarmUpCosineAnnealingLR":
            scheduler = schedulers.WarmUpCosineAnnealingLR(
                optimizer, **scheduler_args)
        elif scheduler_name is None:
            scheduler = None
        else:
            raise NotImplementedError(
                'scheduler not implemented: {}'.format(scheduler_name))

        # return optimizer and scheduler
        if scheduler is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',
                    'interval': scheduler_interval,
                    'frequency': 1
                }
            }

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()


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
            scheduler_hparams: Optional[dict] = None,
            pre_transform_hparams: Optional[dict] = None,
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
            pre_transform_hparams: dict, optional
                Pre-transformation hyperparameters
        """
        if pre_transform_hparams is not None:
            if len(pre_transform_hparams) == 0:
                self.pre_transform = None
            else:
                self.pre_transform = transforms.create_composite_transform(
                    pre_transform_hparams)
                # recompute input and output dimensions
                model_hparams['in_channels'] = self.pre_transform.recompute_indim(
                    model_hparams['in_channels'])
                model_hparams['out_channels'] = self.pre_transform.recompute_outdim(
                    model_hparams['out_channels'])
        else:
            self.pre_transform = None

        super().__init__(
            model, model_hparams, optimizer_hparams, scheduler_hparams)

    def training_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch)
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)

        # apply forward and return log-likelihood and loss
        log_prob = self.model.log_prob(batch)
        loss = -log_prob.mean()

        # log loss and return
        self.log('train_loss', loss, on_epoch=True, on_step=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> FloatTensor:
        batch_size = len(batch)
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)

        # apply forward and return log-likelihood and loss
        log_prob = self.model.log_prob(batch)
        loss = -log_prob.mean()

        # log loss and return
        self.log('val_loss', loss, on_epoch=True, on_step=True, batch_size=batch_size)
        return loss

    def sample(self, *args, **kargs):
        return self.model.sample(*args, **kargs)

    def log_prob(self, *args, **kargs):
        return self.model.log_prob(*args, **kargs)
