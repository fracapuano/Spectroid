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
