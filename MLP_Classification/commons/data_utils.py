import pandas as pd
import os
from fastlangid.langid import LID
from typing import Tuple
from tqdm import tqdm
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

language_identifier = LID()

def load_metadata(path:str="data", clean:bool=True, verbose:int=1)->pd.DataFrame:
    """This function loads the `paper_metadata_mag_mesh.json` file and applies (optional) very basic
    preprocessing steps.
    If verbose, also prints information about the cleaned version of the dataset.

    Args: 
        path (str, optional): Path where to find the `paper_metadata_mag_mesh.json` file. Defaults to `data`.
        clean (bool, optional): Whether or not to apply basic preprocessing such as: 
                                - Removing all papers that are not in english
                                - Removing all papers that do not present Title AND Abstract.
                                Defaults to True.
        verbose (int, optional): Whether or not to print the initial and final number of rows in the cleaned
                                 dataset.
    
    Returns: 
        pd.DataFrame: DataFrame object storing the cleaned version of `paper_metadata_mag_mesh.json`.
    """
    # sanity check
    if "paper_metadata_mag_mesh.json" not in os.listdir(path=path):
        raise ValueError(f"Metadata file not present in path {path} folder!")
    
    # reading the data while keeping the relevant metadata only 
    # (only abstract and title are interesting at this stage)
    scidocs_df = pd.read_json(
    path + "/paper_metadata_mag_mesh.json"
    ).transpose().drop(columns=["paper_id", "year", "authors", "cited_by", "references"])
    
    # storing original len
    original_len = len(scidocs_df)
    # set index name to directly join
    scidocs_df.index.name = "pid"
    
    # retrieving all papers that do not present either title or abstract
    abstract_or_text_mask = scidocs_df.isna().abstract | scidocs_df.isna().title
    # removing invalid papers
    if clean:
        scidocs_df = scidocs_df[~abstract_or_text_mask]
    
    # storing len after having removed papers without title or abstract
    step1_len = len(scidocs_df)
    
    # retrieving all papers that have not an english abstract (assuming english abstract also triggers english title)
    tqdm.pandas(desc="Retrieving non-english papers")
    english_mask = scidocs_df.abstract.progress_apply(lambda abstract: language_identifier.predict(abstract) == "en")
    if clean:
        scidocs_df = scidocs_df[english_mask]

    # storing len after having removed non english papers
    step2_len = len(scidocs_df)

    if verbose > 0:
        print(f"Total number of papers in SciDocs: {original_len}")
        print(f"Total number of papers after data removing abstract/title lacking papers: {step1_len}")
        print(f"Total number of papers after data removing non english papers: {step2_len}")

    return scidocs_df

def load_dataset(path:str="data", dataset:Tuple[str, str]="mesh")->pd.DataFrame:
    """This function returns a dataset correspoding to the union of three partitions of a single one.
    In particular, we here merge the `train`, `val` and `test` partitions.

    Args:
         path (str, optional): Path where to find the `mag`/`mesh` folders. Defaults to `data`.
        dataset (Tuple[str, str], optional): Indicates the dataset to load. Either "mesh" or "mag". 
                                             Defaults to "mesh".

    Returns:
        pd.DataFrame: Merged version of input dataset.
    """
    # sanity checks
    if "mesh" not in os.listdir(path=path) or "mag" not in os.listdir(path=path):
        raise ValueError(f"mag and mesh folders not in {path}!")

    if dataset.lower() not in ["mag", "mesh"]: 
        raise ValueError(f"The only supported dataset are 'mag' and 'mesh'. {dataset} not supported!")

    # concatenating the partitions to form one dataset only.
    df = pd.concat(
        [pd.read_csv(f"{path}/{dataset}/{partition}.csv", index_col=0) for partition in ["train", "val", "test"]]
    )
    return df

def to_hf_dataset(dataset:pd.DataFrame, sep:str="[SEP]")->Dataset:
    """Turns a pd.DataFrame into HF Dataset after having applied custom preprocessing steps
    
    Args: 
        dataset (pd.DataFrame): dataset to turn into HugginFace form.
        sep (str, optional): Separator to use to concatenate title and abstract.
    
    Returns:
        Dataset: HuggingFace dataset with a `text` column, presenting concatenation of title and abstract
                 and a `labels` column, with class labels.
    """
    # turning pandas dataframe to hf instance
    hf = Dataset.from_pandas(dataset)
    # concatenating title and abstract
    text_column = [
        title + sep + abstract 
        for title, abstract in zip(dataset["title"], dataset["abstract"])
    ]
    hf = hf.add_column("text", text_column)

    # rename labels column (as per PyTorch API)
    hf = hf.rename_column("class_label", "labels")
    # turn labels into ClassLabel column
    hf = hf.cast_column("labels", ClassLabel(names=dataset["class_label"].unique().tolist()))
    # remove useless columns
    hf = hf.remove_columns(column_names=["abstract", "title", "pid"])
    return hf

def tokenize_hf(
    hf:Dataset,
    tokenizer:AutoTokenizer,
    batched:bool=True):
    """This function returns a tokenized version of the `hf` dataset using input tokenizer.
    
    Args:
        hf (Dataset): HuggingFace dataset.
        tokenizer (AutoTokenizer): Tokenizer object. Follows HuggingFace API.
        batched (bool, optional): Whether or not to perform tokenization batched.
    """
    def do_tokenize(t:str):
        """Tokenizes input text `t`"""
        tokenized = tokenizer(t, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length=512)
    
        return tokenized
    
    hf_tokenized = hf.map(
    function=do_tokenize, input_columns=["text"], batched=batched
        ).remove_columns("text")
    
    return hf_tokenized

def hf_to_dataloader(hf:Dataset, batch_size:int=8)->DataLoader:
    """This function turns an HuggingFace dataset into a pytorch DataLoader.
    
    Args: 
        hf (Dataset): HuggingFace dataset.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 8.
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # set to torch format converts any list or array to torch tensors
    hf.set_format("torch")
    # builds data
    return DataLoader(hf, batch_size=batch_size)