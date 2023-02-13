import torch
import torch.nn as nn
from transformers import AutoModel
from datasets import Dataset
import wandb
from .model import SpecterClassifier
from .model_utils import Trainer
from torch.optim import AdamW


classification_heads = {
    "MESH":{
        "CH1": [nn.Linear(in_features=768, out_features=11)],
        "CH2": [
            nn.Linear(in_features=768, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=11)
            ],
        "CH2.2": [
            nn.Linear(in_features=768, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=11)
            ],
        "CH3": {"n_layers":5, "n_units":64},
        "project_name": "MFs-MeSH classification"
    },
    "MAG":{
        "CH1": [nn.Linear(in_features=768, out_features=19)],
        "CH2": [
            nn.Linear(in_features=768, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=19)
            ],
        "CH2.2": [
            nn.Linear(in_features=768, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=64), 
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=19)
            ],
        "CH3": {"n_layers":5, "n_units":64},
        "project_name": "MFs-MAG classification"
    }
}

class Experiment:
    def __init__(self, config:dict, dataset:Dataset, models_path:str="trainedmodels", track:bool=True, verbose:int=1): 
        
        self.config = config
        self.dataset = dataset
        self.models_path = models_path
        self.track = track

        # splitting mesh data into training and test data
        self.splits = self.dataset.train_test_split(test_size=self.config["test_size"])

        # input sanity check  
        if config["models_prefix"].startswith("MAG"):
            self.task = "MAG" 
        elif config["models_prefix"].startswith("MESH"):
            self.task = "MESH"
        else: 
            raise ValueError(f"No valid task given in models_prefix (prompted {config['models_prefix'].split('_')[0]})")
        
        if config["models_prefix"].endswith("1"):
            self.head_type = "CH1"
        elif config["models_prefix"].endswith("_2"):  # mag/mesh_2
            self.head_type = "CH2"
        elif config["models_prefix"].endswith("_2.2"):  # mag/mesh_*2.2*
            self.head_type = "CH2.2"
        elif config["models_prefix"].endswith("3"):
            self.head_type = "CH3"
        else:
            raise Warning(f"Only accepted heads are [1, 2, 2.2, 3] (prompted {config['models_prefix'].split('_')[1]})")

        if self.track: 
            # track experiments
            wandb.init(
                # set the wandb project where this run will be logged
                project=classification_heads[self.task]["project_name"],
                # track hyperparameters and run metadata
                config=self.config
            )
        
        # loading pre-trained SPECTER
        self.base_model = AutoModel.from_pretrained("allenai/specter")
        
        # specializing classification head
        self.model = SpecterClassifier(
            base_model=self.base_model, 
            n_labels=11 if self.task == "MESH" else 19,
            n_layers=self.config["hidden_layers"], 
            n_units=self.config["units"],
            use_dropout=config["dropout"],
            use_batchnorm=config["batchnorm"]
            )
        
        if self.head_type != "CH3" and self.head_type is not None:
            self.model.set_classification_head(classification_heads[self.task][self.head_type])

        if verbose>0:
            print("Classification head architecture:")
            print(self.model.classification_head)
            total_params = sum(
                param.numel() for param in self.model.parameters()
            )
            print("Number of parameters (MeSH model): {:.4e}".format(total_params))

    def perform_training(self):
        """Performs training using prompted configuration."""

        # instantiate an optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        # define a classification loss function
        loss_function = torch.nn.CrossEntropyLoss()
        # instantiate a trainer object
        trainer = Trainer(
            model=self.model, 
            splits=self.splits,
            optimizer=optimizer,
            loss_function=loss_function,
            batch_size=self.config["batch_size"]
            )
        # perform training
        trainer.do_train(n_epochs=self.config["epochs"], log_every=5, models_prefix=self.config["models_prefix"])
        # saves last model
        torch.save(self.model.state_dict(), self.models_path + f"/{self.task}_{self.head_type}.pth")
        if self.track: # stops wandb
            wandb.finish()
        
    def load_run(self, model_path:str=None, checkpoint:bool=False, epoch:int=None): 
        """Loads a pre-trained model given the considered configuration
        
        Args: 
            model_path (str, optional): Directory where to find the model considered.
            checkpoint (bool, optional): When True, considers models in the `checkpoints` folder, otherwise downlods 
                                         from `trainedmodels`. Defaults to True.
            epoch (int, optional): The epoch to download. Can be not None only when checkpoint is True. Defaults to 
                                   None.
        """
        # sanity check
        if not checkpoint and epoch is not None: 
            print(f"Checkpoint:{checkpoint} - epoch:{epoch}")
            raise ValueError("Epoch can be not None only when checkpoint is True")
        
        if checkpoint:
            path = "checkpoints/" if model_path is None else model_path
            model_name = f"{self.models_prefix}epoch_{epoch}.pth"
        elif not checkpoint or checkpoint is None:
            path = "trainedmodels/" if model_path is None else model_path
            model_name=f"{self.task}_{self.head_type}.pth"
        
        # loads model
        self.model.load_state_dict(torch.load(path + model_name))
        print(f"Model {model_path} loaded successfully!")

    def test_model(self):
        """Performs testing."""
        # instantiate an optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        # define a classification loss function
        loss_function = torch.nn.CrossEntropyLoss()
        
        tester= Trainer(
            model=self.model,
            splits=self.splits,
            optimizer=optimizer, 
            loss_function=loss_function,
            batch_size=32
        )

        avg_f1 = tester.do_test()
        
        print("\nAverage F1-Score {:.4f}".format(avg_f1))
