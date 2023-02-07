"""
Mainly implements a Trainer object used to interface more easily the training process.
Other utils are also defined.
"""
import torch
from torch.optim import Optimizer
from datasets import Dataset, DatasetDict
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import wandb
from .data_utils import hf_to_dataloader
from rich.progress import track

class Trainer: 
    def __init__(
        self,
        model:torch.nn.Module, 
        splits:DatasetDict, 
        optimizer:Optimizer,
        loss_function:_WeightedLoss,
        batch_size:int=4,
        use_gpu:bool=True):
        """Trainer interface."""

        self.train_loader = DataLoader(
            splits["train"], 
            batch_size=batch_size, 
            shuffle=True
            )

        self.test_loader = DataLoader(
            splits["test"], 
            batch_size=batch_size, 
            shuffle=True
            )
        
        self.optimizer = optimizer
        self.loss = loss_function

        self.model = model
        self.batch_size = batch_size
        
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    
    def do_train(self, n_epochs:int=5, log_every:int=10, models_prefix:str=""): 
        """
        Performs training.

        Args:
            n_epochs (int, optional): Training epochs. Defaults to 5.
            log_every (int, optional): How often (number of batches) to log
                                       current training loss. Defaults to 10.
            models_prefix (str, optional): Prefix to model names. Defaults to "".
        """
        device = self.device
        
        # set training mode
        self.model.train()
        step = 0
        for epoch in (training_bar:=tqdm(range(n_epochs))): 
            # loop over training data
            for batch in self.train_loader:
                labels = batch["labels"].to(device)
                # zerograd optimizer
                self.optimizer.zero_grad()
                # forward pass
                outputs = self.model(batch)
                # loss computation
                loss_value = self.loss(outputs, labels)
                # backward pass
                loss_value.backward()
                # step parameters
                self.optimizer.step()
                
                training_bar.set_description("Training Loss: {:.4f}".format(loss_value.item()))
                step += 1

                if step % log_every == 0: 
                    training_bar.set_description("Training Loss: {:.4f}".format(loss_value.item()))
                    wandb.log({
                        "CrossEntropy-TrainingLoss": loss_value.item(),
                        "MacroF1-Val": self.do_test(tqdm_mute=True)
                    })
            
            # saving the model at each epoch - for diagnostics purposes
            model_name = "checkpoints/" + models_prefix + f"epoch_{epoch}.pth"
            torch.save(self.model.cpu().state_dict(), model_name)

    def do_test(self, tqdm_mute:bool=False)->float: 
        """Performs testing for the considered model. Tests f1 score in particular.
        
        Args: 
            tqdm_mute (bool, optional): Whether or not to show a progress bar iterating during the testing phase.
        
        Returns: 
            float: Macro-Average f1 score over all batches in test set.
        """
        
        # set testing mode
        self.model.eval()

        progress_bar = tqdm(self.test_loader) if not tqdm_mute else self.test_loader

        batches_f1 = torch.zeros(len(self.test_loader))
        with torch.no_grad():
            idx = 0
            for batch in progress_bar:
                output = self.model(batch).cpu()
                y_true = batch["labels"]
                # max(axis=1) gives the maximal value and maximal index too
                _, y_batch = torch.max(output, 1)
                # store this batch f1 score
                batches_f1[idx] = f1_score(
                    y_true=y_true, 
                    y_pred=y_batch.numpy(),
                    average="macro")
                
                idx += 1
        
        return batches_f1.mean().item()
    
def embed_data(model:torch.nn.Module, data:Dataset, batch_size:int=32)->torch.Tensor:
    """Forwards the data presented in data and returns an array of the correspoding embeddings
    
    Args: 
        model (torch.nn.Module): model to use to embed the papers considered
        data (Dataset): Dataset to use containing the data to embed.
        batch_size (int, optional): Batch size for the DataLoader built on top of `data`.
    
    Returns:
        torch.Tensor: tensor containing the embedded papers.
    """
    dataloader = hf_to_dataloader(hf=data, batch_size=batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        embeddings = []
        for batch in tqdm(dataloader, desc="Obtaining embeddings"):
            # moving batch of data on target device
            batch = {k: v.to(device) for k, v in batch.items()}
            # forwarding batch of data through the considered network
            outputs = model(**batch)
            # appending the embeddings to the list of considered ones
            embeddings.append(outputs.last_hidden_state[:,0,:].to("cpu"))

            del batch
            torch.cuda.empty_cache()
    
    return torch.cat(embeddings)
