
import configparser
import os

# read in environmental variables from config file
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "envs.ini"))

# set environmental variables
DEFAULT_DATASETS_DIR = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_DATASETS_DIR")
DEFAULT_GALAXIES_DIR = config.get(
    "ENVIRONMENT_VARIABLES", "DEFAULT_GALAXIES_DIR")

# make sure the directories exist
os.makedirs(DEFAULT_DATASETS_DIR, exist_ok=True)
os.makedirs(DEFAULT_GALAXIES_DIR, exist_ok=True)
