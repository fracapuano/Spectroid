import json
from tqdm import tqdm
import os
import random
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def get_metadata(data_file:str):
    """Instead of loading the entire json in memory, scans through the lines one at the time.

    Args:
        data_file (str): path to the arxiv metadata json file

    Yields:
        str: non-parsed dictionary, contains one arxiv paper and its metadata.
    """
    with open(data_file, 'r') as f:
        for line in f:
            yield line

def create_arxiv(output_dir:str, n_papers:int=50_000, train_size:float=0.7, val_size:float=0.15, test_size:float=0.15):
    """Create data.json and metadata.json as per SPECTER specifications using arXiv dataset.
    The original dataset can be retrieved either using a locally-saved json, or by interacting with the Kaggle API.
    The Kaggle API functionality is not supported on Windows.

    Args:
        output_dir (str): path where to save the files
        n_papers (int, optional): number of papers to consider in the subset. Defaults to 50_000.

    Raises:
        ValueError: one (and only one) option between using a local version of the arXiv dataset and using the Kaggle API should be selected.
    """
    print('Connecting to Kaggle...')
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('Cornell-University/arxiv', path = f'{output_dir}/arxiv_data', unzip=True)
    
    arxiv_metadata = get_metadata(data_file=f'{output_dir}/arxiv_data/arxiv-metadata-oai-snapshot.json')
   
    # initialise counter: number of papers in the subset
    papers_in_subset = 0
    # initialise empty dictionary -> metadata.json will be built starting from this dictionary
    metadata = {}
    # create another dictionary, where the key is the paper_id and the value is the label (i.e., the category)
    labels = {}
    # iterate until you add `n_papers` papers in your subset
    with tqdm(total=n_papers) as pbar:
        for paper in arxiv_metadata:
            # load paper_information
            paper_dict = json.loads(paper)
            # try is necessary because not all fields are always defined
            try:
                # we only want to consider papers which belong to one and only one category
                # this should help obtaining us more reliable results for classification
                if len(paper_dict.get('categories').split(" ")) == 1:
                    # each paper's category is in the form category.topic (e.g., math.CO)
                    # we are restricting our analysis to the top-6 categories
                    if paper_dict.get('categories').split(".")[0] in ["astro-ph", "cs", "math", "physics", "q-bio", "stat"]:
                        # obtain paper_id. paper_id is slightly modified to make it more similar to Scidocs format 
                        paper_id = paper_dict.get('id').replace('.', '')[1:]
                        # add this papers to metadata.json
                        metadata[paper_id] = {
                            'paper_id' : paper_id, 
                            'title' : paper_dict.get('title'), 
                            'abstract' : paper_dict.get('abstract'), 
                        }
                        # add both category and topic to the labels dataframe
                        labels[paper_id] = {
                            'topic' : paper_dict.get('categories').split(".")[0],
                            'subtopic' : paper_dict.get('categories')
                            }
                        # increment number of papers in the subset
                        papers_in_subset += 1
                        pbar.update(1)
                        # subset is ready
                        if papers_in_subset == n_papers:
                            # store metadata
                            with open(f'{output_dir}/arxiv_data/metadata.json', 'w') as f:
                                json.dump(metadata, f)
                            # store the labels in 70/15/15 train/val/test split
                            # get the paper_ids 
                            paper_keys = list(labels.keys())
                            # store them
                            with open(f'{output_dir}/arxiv_data/arxiv.ids', 'w') as f:
                                for key in paper_keys:
                                    f.write(key + '\n')
                            # randomly split them
                            random.shuffle(paper_keys)
                            train_papers = int(train_size * len(paper_keys))
                            val_papers = int(val_size * len(paper_keys))
                            # create 3 dictionaries with the given proportions
                            train = {key: labels[key] for key in paper_keys[:train_papers]}
                            val = {key: labels[key] for key in paper_keys[train_papers:train_papers+val_papers]}
                            test = {key: labels[key] for key in paper_keys[train_papers+val_papers:]}
                            os.makedirs(f'{output_dir}/arxiv_data/classification')
                            # store the labels
                            pd.DataFrame(train).T["topic"].to_csv(f'{output_dir}/arxiv_data/classification/train.csv')
                            pd.DataFrame(val).T["topic"].to_csv(f'{output_dir}/arxiv_data/classification/val.csv')
                            pd.DataFrame(test).T["topic"].to_csv(f'{output_dir}/arxiv_data/classification/test.csv')
                            break
            except:
                pass 

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', help='directory where the user wishes to store the files', default = 'testing_datasets')
    ap.add_argument('--train-size', help='size of train split', type=restricted_float, default=0.7)
    ap.add_argument('--val-size', help='size of val split', type=restricted_float, default=0.15)
    ap.add_argument('--n_papers', help='number of papers to consider in the subset', type=int, default=50_000)
        
    args = ap.parse_args()
    train_size = args.train_size
    val_size = args.val_size
    test_size = 1 - train_size - val_size

    create_arxiv(output_dir=args.output_dir, n_papers = args.n_papers, train_size = train_size, val_size = val_size, test_size = test_size)
