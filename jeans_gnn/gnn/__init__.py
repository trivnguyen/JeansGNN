
from . import modules, graph_regressors

def create_graph_regression_module(config):
    """ Create a graph regressor module from config file """

    model = modules.MAFModule(
        model=graph_regressors.GraphRegressor,
        transform=None,
        model_hparams=config["model"]["parameters"],
        transform_hparams=config["transform"]["parameters"],
        optimizer_hparams=config["optimizer"]["parameters"]
    )