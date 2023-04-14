
import logging
import os
import sys
from typing import List, Optional, Tuple, Union

import yaml

import pytorch_lightning as pl
import torch
import utils
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from .gnn import graph_regressors, transforms


class DensitySampler():
    """ Sample the dark matter density from kinematic data

    Parameters
    ----------
    run_name: str
        Name of the run
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
    run_prefix: str
        Prefix of the run directory

    Attributes
    ----------
    run_name: str
        Name of the run
    run_prefix: str
        Prefix of the run directory
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
    output_dir: str
        Output directory
    logger: logging.Logger
        Logger

    Methods
    -------
    create_dataloader(
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        flag: str = 'train',
        verbose: bool = True,
        **kwargs
    ):
        Create a data loader from a dataset

    fit(
        train_dataset_name: Optional[str] = None,
        train_dataset_path: Optional[str] = None,
        val_dataset_name: Optional[str] = None,
        val_dataset_path: Optional[str] = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 1,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        **kwargs
    ):
    """

    def __init__(
            self,
            run_name: str,
            model_type: str = 'GNN',
            model_params: Optional[dict] = None,
            optimizer_params: Optional[dict] = None,
            scheduler_params: Optional[dict] = None,
            transform_params: Optional[dict] = None,
            run_prefix: Optional[str] = None,
        ):

        if model_params is None:
            model_params = {}
        if optimizer_params is None:
            optimizer_params = {}
        if scheduler_params is None:
            scheduler_params = {}
        if transform_params is None:
            transform_params = {}

        self.run_name = run_name
        self.run_prefix = run_prefix
        self.model_type = model_type
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.transform_params = transform_params

        # set up logger, model, and output directory
        self._setup_logger()
        self._setup_model()
        self._setup_dir()

    # create output directory and write all params into yaml
    def _setup_dir(self):
        """ Set up the output directory and write all params into yaml """
        # create an output directory
        self.output_dir = os.path.join(self.run_prefix, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # write all params into yaml
        params = {
            'run_name': self.run_name,
            'run_prefix': self.run_prefix,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'optimizer_params': self.optimizer_params,
            'scheduler_params': self.scheduler_params,
            'transform_params': self.transform_params,
        }
        with open(os.path.join(self.output_dir, 'params.yaml'), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    def _setup_logger(self):
        """ Set up logger """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _setup_model(self):
        """ Set up model and transformation """
        if self.model_type == 'GNN':
            self.model = graph_regressors.GraphRegressorModule(
                model_hparams=self.model_params,
                optimizer_hparams=self.optimizer_params,
                scheduler_hparams=self.scheduler_params)
            self.transform = transforms.PhaseSpaceGraphProcessor(
                **self.transform_params)
        elif self.model_type == 'Jeans':
            self.model = None
            self.transform = lambda x: x
        else:
            raise NotImplementedError(
                f"Model {self.model_type} not implemented")

    @static
    def load_from_dir(
            run_dir: str,
            model_type: Optional[str] = None,
            model_params: Optional[dict] = None,
            optimizer_params: Optional[dict] = None,
            scheduler_params: Optional[dict] = None,
            transform_params: Optional[dict] = None,
        ):
        """ Load a DensitySampler from a directory

        Parameters
        ----------
        run_dir: str
            Directory of the run
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
        sampler: DensitySampler
        """
        # load params from yaml
        with open(os.path.join(run_dir, 'params.yaml'), 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        # overwrite params if provided
        if model_type is not None:
            params['model_type'] = model_type
        if model_params is not None:
            params['model_params'] = model_params
        if optimizer_params is not None:
            params['optimizer_params'] = optimizer_params
        if scheduler_params is not None:
            params['scheduler_params'] = scheduler_params
        if transform_params is not None:
            params['transform_params'] = transform_params

        # create a DensitySampler
        sampler = DensitySampler(**params)
        return sampler

    def create_dataloader(
            self,
            dataset_name: Optional[str] = None,
            dataset_path: Optional[str] = None,
            flag: str = 'train',
            verbose: bool = True,
            **kwargs
        ):
        """ Create a data loader from a dataset

        Parameters
        ----------
        dataset_name: str
            Name of the dataset
        dataset_path: str
            Path to the dataset. Ignored if dataset_name is provided
        flag: str
            Flag of the dataset. Only used if dataset_name is provided
        verbose: bool
            Whether to print out the dataset information
        kwargs: dict
            Keyword arguments for DataLoader
        Returns
        -------
        dataloader: torch_geometric.loader.DataLoader
        """
        # find dataset path, return None if not found
        if dataset_name is not None:
            path = utils.get_dataset_path(dataset_name, flag=flag)
            if path is None:
                self.logger.info(f"Dataset {dataset_name} not found. Return None.")
                return None
        elif dataset_path is not None:
            path = dataset_path
            if not os.path.exists(path):
                self.logger.info(f"Dataset {path} not found. Return None.")
                return None
        else:
            self.logger.info("No dataset provided. Return None.")
            return None

        # read the dataset
        node_features, graph_features, headers = utils.read_graph_dataset(
            path, features_list=['pos', 'vel', 'labels'])

        # print out dataset information
        if verbose:
            self.logger.info(f"Dataset: {path}")
            self.logger.info(f"Number of graphs: {len(node_features)}")
            self.logger.info("Headers:")
            for header in headers:
                self.logger.info(f"{header}: {headers[header]}")

        # create a graph dataset
        dataset = []
        for i in range(len(node_features)):
            pos = node_features[i]['pos']
            vel = node_features[i]['vel']
            labels = graph_features[i]['labels']
            graph = self.transforms(pos, vel, labels)
            dataset.append(graph)

        # create a data loader
        return DataLoader(dataset, **kwargs)

    def fit(
            self,
            dataset_name: Optional[str] = None,
            train_dataset_path: Optional[str] = None,
            val_dataset_path: Optional[str] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            batch_size: int = 1,
            num_workers: int = 1,
            max_epochs: int = 100,
            min_delta: float = 0.0,
            patience: int = 20,
        ):
        """ Fit the model

        Parameters
        ----------
        dataset_name: str
            Name of the dataset
        train_dataset_path: str
            Path to the training dataset. Ignored if dataset_name is provided
        val_dataset_path: str
            Path to the validation dataset. Ignored if dataset_name is provided
        train_loader: torch_geometric.loader.DataLoader
            Training data loader. Ignored if dataset_name or train_dataset_path
            is provided
        val_loader: torch_geometric.loader.DataLoader
            Validation data loader. Ignored if dataset_name or val_dataset_path
            is provided
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
        """
        # Create data loaders
        pin_memory = True if torch.cuda.is_available() else False
        if train_loader is None:
            train_loader = self.create_dataloader(
                dataset_name=dataset_name, dataset_path=train_dataset_path,
                flag='train', batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory)
        if val_loader is None:
            val_loader = self.create_dataloader(
                dataset_name=dataset_name, dataset_path=val_dataset_path,
                flag='val', batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory)
        if val_loader is None:
            self.logger.info(
                "Validation dataset not found. Will not perform validation")

        # Create a trainer
        # set up callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}", save_weights_only=True,
                mode="min", monitor="val_loss"),
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
            logger=CSVLogger(
                self.output_dir, name=self.run_name, version=0),
            callbacks=callbacks
        )

        # fit the model
        trainer.fit(self.model, train_loader, val_loader)

    def sample(self):
        raise NotImplementedError
