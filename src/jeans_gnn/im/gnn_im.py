
import os
import glob
from typing import List, Optional, Tuple, Union
import logging
import shutil

logger = logging.getLogger(__name__)

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import DataLoader

from .. import utils
from ..gnn import graph_regressors, transforms

class GNNInferenceModel():
    """ Sample the dark matter density from kinematic data

    Attributes
    ----------
    run_name: str
        Name of the run
    run_prefix: str
        Prefix of the run
    model_params: dict
        Parameters for the graph model
    optimizer_params: dict
        Parameters for the optimizer
    scheduler_params: dict
        Parameters for the scheduler
    transform_params: dict
        Parameters for the transforms

    Methods
    -------
    _setup_model()
        Set up the model
    _setup_dir()
        Set up the output directory

    """

    def __init__(
            self,
            run_name: str,
            config_file: Optional[str] = None,
            model_params: Optional[dict] = None,
            optimizer_params: Optional[dict] = None,
            scheduler_params: Optional[dict] = None,
            transform_params: Optional[dict] = None,
            run_prefix: Optional[str] = None,
            resume: bool = False,
        ):
        """

        Parameters
        ----------
        run_name: str
            Name of the run
        config_file: str
            Path to the config file
        model_params: dict
            Parameters for the graph model. Overwrites config file `model` if given
        optimizer_params: dict
            Parameters for the optimizer. Overwrites config file `optimizer` if given
        scheduler_params: dict
            Parameters for the scheduler. Overwrites config file `scheduler` if given
        transform_params: dict
            Parameters for the transforms. Overwrites config file `transform` if given
        run_prefix: str
            Prefix of the run
        resume: bool
            Resume from a previous run
        """
        # set default values
        if model_params is None:
            model_params = {}
        if optimizer_params is None:
            optimizer_params = {}
        if scheduler_params is None:
            scheduler_params = {}
        if transform_params is None:
            transform_params = {}
        if run_prefix is None:
            run_prefix = ''

        # read config file
        if config_file is not None:
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.model_params = config['model']
            self.optimizer_params = config['optimizer']
            self.scheduler_params = config['scheduler']
            self.transform_params = config['transform']
            # overwrite config file with given params
            self.model_params.update(model_params)
            self.optimizer_params.update(optimizer_params)
            self.scheduler_params.update(scheduler_params)
            self.transform_params.update(transform_params)
        else:
            self.model_params = model_params
            self.optimizer_params = optimizer_params
            self.scheduler_params = scheduler_params
            self.transform_params = transform_params

        self.run_name = run_name
        self.run_prefix = run_prefix
        self.output_dir = None
        self.model = None
        self.transform = None

        # set up logger, model, and output directory
        self._setup_dir(resume=resume)
        self._setup_model(resume=resume)

    # create output directory and write all params into yaml
    def _setup_dir(self, resume: bool = False):
        """ Set up the output directory and write all params into yaml """
        # create an output directory
        # overwrite existing directory if not resuming
        self.output_dir = os.path.join(self.run_prefix, self.run_name)
        if not resume:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # write all params into yaml
        params = {
            'run_name': self.run_name,
            'run_prefix': self.run_prefix,
            'model_params': self.model_params,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'transform_params': self.transform_params,
        }
        with open(
            os.path.join(self.output_dir, 'params.yaml'),
            'w', encoding='utf-8') as f:
            yaml.dump(params, f, default_flow_style=False)

    def _setup_model(self, resume: bool = False):
        """ Set up model and transformation """
        if not resume:
            self.model = graph_regressors.GraphRegressorModule(
                model_hparams=self.model_params,
                optimizer_hparams=self.optimizer_params,
                scheduler_hparams=self.scheduler_params)
        else:
            checkpoint = self._find_best_checkpoint()
            self.model = graph_regressors.GraphRegressorModule.load_from_checkpoint(
                checkpoint_path=checkpoint,
                model_hparams=self.model_params,
                optimizer_hparams=self.optimizer_params,
                scheduler_hparams=self.scheduler_params
            )
        self.transform = transforms.PhaseSpaceGraphProcessor(
            **self.transform_params)

    def _find_best_checkpoint(self):
        """ Find the best checkpoint

        Returns
        -------
        str
            Path to the best checkpoint
        """
        # find all checkpoints
        checkpoints = os.path.join(
            self.output_dir, 'lightning_log/checkpoints', 'epoch=*.ckpt')
        checkpoints = sorted(glob.glob(checkpoints))

        # find the best checkpoint
        best_checkpoint = None
        best_loss = 1e10
        for checkpoint in checkpoints:
            loss = float(checkpoint.split('=')[-1].split('.')[0])
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint
        return best_checkpoint

    def fit(
            self,
            train_dataset_path: Optional[str] = None,
            val_dataset_path: Optional[str] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            max_epochs: int = 100,
            min_delta: float = 0.0,
            patience: int = 20,
            save_top_k: int = 2,
        ):
        """ Fit the model

        Parameters
        ----------
        train_loader: torch_geometric.loader.DataLoader
            Training data loader
            is provided
        val_loader: torch_geometric.loader.DataLoader
            Validation data loader
        train_dataset_path: str
            Path to the training dataset. Ignore if `train_loader` is provided
        val_dataset_path: str
            Path to the validation dataset. Ignore if `val_loader` is provided
        batch_size: int
            Batch size
        num_workers: int
            Number of workers for data loading
        max_epochs: int
            Maximum number of epochs
        min_delta: float
            Minimum delta for early stopping
        patience: int
            Patience for early stopping
        save_top_k: int
            Number of best checkpoints to save
        """
        # Create data loaders
        pin_memory = True if torch.cuda.is_available() else False
        if train_loader is None:
            if train_dataset_path is None:
                raise ValueError(
                    "Either `train_loader` or `train_dataset_path` "
                    "must be provided")
            train_loader = utils.dataset.create_dataloader(
                train_dataset_path, self.transform,
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory)

        if val_loader is None:
            if val_dataset_path is None:
                logger.info(
                    "Validation dataset not found. Will not perform validation")
            else:
                val_loader = utils.dataset.create_dataloader(
                    val_dataset_path, self.transform,
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory)

        # Create a trainer
        # set up callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}", save_weights_only=False,
                mode="min", monitor="val_loss", save_top_k=save_top_k),
            pl.callbacks.LearningRateMonitor("epoch")
        ]
        # add early stopping if validation dataset is available
        if val_loader is not None:
            callbacks.append(
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor="val_loss",
                    min_delta=min_delta,
                    patience=patience,
                    mode="min", verbose=True))
        trainer = pl.Trainer(
            default_root_dir=self.output_dir,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else 0,
            max_epochs=max_epochs,
            logger=CSVLogger(self.output_dir, name='lightning_log', version=''),
            callbacks=callbacks
        )

        # fit the model
        trainer.fit(self.model, train_loader, val_loader)

    def sample(
            self,
            num_samples: int,
            data_loader: Optional[DataLoader] = None,
            dataset_path: Optional[str] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            device: Optional[torch.device] = None,
            return_labels: bool = False,
            to_numpy: bool = True,
        ):
        """ Sample the posterior distribution

        Parameters
        ----------
        num_samples: int
            Number of samples to generate
        data_loader: torch_geometric.loader.DataLoader
            Data loader
        dataset_path: str
            Path to the dataset. Ignore if `data_loader` is provided
        batch_size: int
            Batch size
        num_workers: int
            Number of workers for data loading
        device: torch.device
            Device to use. If None, use the device of the model
        return_labels: bool
            Whether to return the labels
        to_numpy: bool
            Whether to convert the samples and labels to numpy arrays

        Returns
        -------
        posteriors: torch.Tensor or np.ndarray
            Samples from the posterior distribution
        labels: torch.Tensor or np.ndarray
            Labels of the samples
        """
        # Set device
        if device is None:
            device = self.model.device
        self.model.to(device)

        # Create data loader
        pin_memory = True if torch.cuda.is_available() else False
        if data_loader is None:
            if dataset_path is None:
                raise ValueError(
                    "Either `data_loader` or `dataset_path` "
                    "must be provided")
            data_loader = utils.dataset.create_dataloader(
                dataset_path, self.transform,
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory)

        # Make sure the model is in eval mode
        self.model.eval()
        with torch.no_grad():
            # Iterate through the data loader
            posteriors = []
            labels = []
            for batch in data_loader:
                batch = batch.to(device)
                posterior = self.model.sample(
                    batch, num_samples=num_samples)
                posteriors.append(posterior)
                if return_labels:
                    labels.append(batch.y)
        posteriors = torch.cat(posteriors, dim=0)

        if return_labels:
            labels = torch.cat(labels, dim=0)
            if to_numpy:
                labels = labels.detach().numpy()
                posteriors = posteriors.detach().numpy()
            return posteriors, labels
        else:
            if to_numpy:
                posteriors = posteriors.detach().numpy()
            return posteriors

    @staticmethod
    def load_from_dir(
            run_dir: Optional[str] = None,
            run_name: Optional[str] = None,
            run_prefix: Optional[str] = None,
            config_file: Optional[str] = None,
            model_params: Optional[dict] = None,
            optimizer_params: Optional[dict] = None,
            scheduler_params: Optional[dict] = None,
            transform_params: Optional[dict] = None,
        ):
        """ Load a Inference Model from a directory

        Parameters
        ----------
        run_dir: str
            Directory of the run
        run_name: str
            Name of the run. Ignored if run_dir is provided
        run_prefix: str
            Prefix of the run. Ignored if run_dir is provided
        model_type: str
            Type of the model. Currently only 'GNN' is supported
        model_params: dict
            Parameters for the model
        optimizer_params: dict
            Parameters for the optimizer
        scheduler_params: dict
            Parameters for the scheduler
        transform_params: dict
            Parameters for the transformation

        Returns
        -------
        sampler: Inference Model
        """
        if run_name is not None:
            if run_prefix is None:
                run_prefix = ''
            # overwrite run_dir if provided
            run_dir = os.path.join(run_prefix, run_name)

        # check if the directory exists
        logger.info(f"Reloading prior run from {run_dir}")
        if not os.path.exists(run_dir):
            raise ValueError(f"Directory {run_dir} does not exist")

        # load params from yaml
        with open(os.path.join(run_dir, 'params.yaml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        # overwrite params if provided
        if model_params is not None:
            params['model_params'].update(model_params)
        if optimizer_params is not None:
            params['optimizer_params'].update(optimizer_params)
        if scheduler_params is not None:
            params['scheduler_params'].update(scheduler_params)
        if transform_params is not None:
            params['transform_params'].update(transform_params)
        params['config_file'] = config_file

        # create a Inference Model
        sampler = GNNInferenceModel(resume=True, **params)

        return sampler