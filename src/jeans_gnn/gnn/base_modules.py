
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
        model_hparams = model_hparams or {}
        optimizer_hparams = optimizer_hparams or {}
        scheduler_hparams = scheduler_hparams or {}
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.save_hyperparameters()

        # init model, transformation, and optimizer
        self.model = model(**model_hparams)

    def forward(self, x, *args, **kargs):
        """ Forward pass """
        return self.model(x, *args, **kargs)


    def configure_optimizers(optimizer_args, scheduler_args=None):
        """ Return optimizer and scheduler for Pytorch Lightning """
        scheduler_args = scheduler_args or {}

        # setup the optimizer
        if optimizer_args['type'] == "Adam":
            return torch.optim.AdamW(
                parameters(),
                lr=optimizer_args.get('lr', 1e-3),
                betas=optimizer_args.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_args.get('weight_decay', 0.0).
                eps=optimizer_args.get('eps', 1e-8)
            )
        elif optimizer_args['type'] == "AdamW":
            return torch.optim.AdamW(
                parameters(),
                lr=optimizer_args.get('lr', 1e-3),
                betas=optimizer_args.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_args.get('weight_decay', 0.01).
                eps=optimizer_args.get('eps', 1e-8)
            )
        else:
            raise NotImplementedError(
                "Optimizer {} not implemented".format(optimizer_args.get('type')))

        # setup the scheduler
        if scheduler_args.get('type') is None:
            scheduler = None
        elif scheduler_args.get('type') == 'ReduceLROnPlateau':
            scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min',
                factor=scheduler_args.get('factor', 0.1),
                patience=scheduler_args.get('patience', 10),
            )
        elif scheduler_args.get('type') == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_args.get('T_max', 100),
                eta_min=scheduler_args.get('eta_min', 0.0)
            )
        elif scheduler_args.get('type') == 'WarmUpCosineAnnealingLR':
            scheduler = models_utils.WarmUpCosineAnnealingLR(
                optimizer,
                decay_steps=scheduler_args.get('decay_steps', 100),
                warmup_steps=scheduler_args.get('warmup_steps', 10),
                eta_min=scheduler_args.get('eta_min', 0.0)
            )
        else:
            raise NotImplementedError(
                "Scheduler {} not implemented".format(self.scheduler_args.get('type')))

        if scheduler is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss',
                    'interval': scheduler_args.get('interval', 'epoch'),
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
