import torch.nn as nn
import torch
from copy import deepcopy as copy

class SpecterClassifier(nn.Module):
    def __init__(
        self, 
        base_model, 
        n_labels:int, 
        n_layers:int=2, 
        n_units:int=64,
        activation_function:str="relu", 
        use_gpu:bool=True,
        use_dropout:bool=False,
        use_batchnorm:bool=False): 
        """
        Args: 
            base_model (transformers.AutoModel): Pre-trained model from the trans-
                                                 formers API.
            n_labels (int): Number of labels that need to predicted
            n_layers (int, optional): Number of layers in the classification head. 
                                      Defaults to 2.
            n_units (int, optional): Number of units in each hidden layer of the
                                     classification head. Defaults to 64.
            activation_function (str, optional): Activation function to be used in the
                                                 classification head. Defaults to ReLU.
            use_gpu (bool, optional): Whether or not to use an available GPU. Defaults to True.
            use_dropout(bool, optional): Whether or not to use dropout to regularize the classification
                                         head of the network. Defaults to False.
            use_batchnorm(bool, optional): Whether or not to use 1d batch norm to scale layers.
                                           Defaults to False.
        """
        super().__init__()

        # specter model
        self.model = copy(base_model)

        # accessing output dimension of base_model
        *_, prelast, _ = self.model.modules()

        transformer_output = nn.Linear(prelast.out_features, n_units)
        hidden_layers = [nn.Linear(n_units, n_units) for _ in range(n_layers)]
        logits_layer = nn.Linear(n_units, n_labels)
        
        if activation_function.lower()=="relu": 
            self.act_func = nn.ReLU
        elif activation_function.lower()=="tanh":
            self.act_func = nn.Tanh
        elif activation_function.lower()=="sigmoid":
            self.act_func = nn.Sigmoid
        else:
            print(f"Input Activation function: {activation_function}")
            raise NotImplementedError("No activation functions other than ReLU currently implemented")
        
        # classification head is a cell-like structure defined as
        # (layer->act_function->layer...) x n_layers
        layers = [transformer_output, *hidden_layers, logits_layer]
        
        act_functions = [self.act_func() for _ in range(len(layers)-1)]
        
        clf_head = [None for _ in range(len(layers) + len(act_functions))]
        clf_head[::2] = layers; clf_head[1::2] = act_functions
        
        if use_batchnorm:
            for b_index in range(len(clf_head)-1, 3):
                clf_head.insert(b_index, nn.BatchNorm1d(num_features=n_units))
        
        if use_dropout:
            for d_index in range(len(clf_head)-1, 4):
                clf_head.insert(d_index, nn.Dropout(p=0.5))

        self.classification_head = nn.Sequential(*clf_head)
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    def set_classification_head(self, clf_head:nn.Sequential):
        """Sets a classification head for the model considered"""
        if isinstance(clf_head, nn.Sequential):
            self.classification_head = clf_head
        elif isinstance(clf_head, list): 
            self.classification_head = nn.Sequential(*clf_head)
        else: 
            raise ValueError("Classification Head not a list nor an iterable!")

    def forward(self, x:dict)->torch.Tensor:
        """Forward pass"""
        device = self.device
        specter_input = {
            key: x[key].to(device) for key in ["input_ids", "token_type_ids", "attention_mask"]
        }
        specter_model = self.model.to(device)
        specter_output = specter_model(**specter_input)
        # remove un-necessary input from device memory
        del specter_input
        # classification is applied on specter output
        classifier_input = specter_output[1]
        classifier_model = self.classification_head.to(device)
        return classifier_model(classifier_input)
