from pathlib import Path
import os
import random
import numpy as np 
import torch

seed = 777

def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent

def seedBasic(seed:int=seed):
    """sets random seed for basic libraries"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
def seedTorch(seed:int=seed):
    """sets random seed for torch & cuda"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + torch 
def seedEverything(seed:int=seed):
    """sets random seed for everything"""
    seedBasic(seed)
    seedTorch(seed)

mesh_config_1 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "1-layer only",
    "hidden_layers": 1,
    "units": 1,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MESH_1", 
    "batch_size":8
    }

mesh_config_2 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "2-layers only, with RELU",
    "hidden_layers": 2,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MESH_2", 
    "batch_size":8
    }

mesh_config_3 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "5-layers, with RELU",
    "hidden_layers": 5,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MESH_3", 
    "batch_size":8
    }

mesh_config_2bis = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "3-layers, with RELU",
    "hidden_layers": 3,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MESH_2.2",
    "batch_size":32  # requires 24+ GB GPU
    }

mag_config_1 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "1-layer only",
    "hidden_layers": 1,
    "units": 1,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MAG_1", 
    "batch_size":8
    }

mag_config_2 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "2-layers only, with RELU",
    "hidden_layers": 2,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MAG_2", 
    "batch_size":8
    }

mag_config_3 = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "5-layers, with RELU",
    "hidden_layers": 5,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MAG_3",
    "batch_size":8
    }

mag_config_2bis = {
    "learning_rate": 5e-5,
    "test_size":0.1,
    "architecture": "3-layers, with RELU",
    "hidden_layers": 3,
    "units": 64,
    "epochs": 5,
    "dropout":False,
    "batchnorm":False,
    "models_prefix": "MAG_2.2",
    "batch_size":32
    }