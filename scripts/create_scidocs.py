import json
from fastlangid.langid import LID
from typing import Iterator
from tqdm import tqdm
import os
import argparse
import random

def split_data(data, train_file, val_file, test_file, train_size, val_size, test_size):
    """Given a list, saves three files train.txt val.txt test.txt with arbitrary random split.

    Args:
        data: list to split
        train_file: path where to store the train.txt file
        val_file: path where to store the val.txt file
        test_file: path where to store the test.txt file
    """
    # random shuffle
    random.shuffle(data)
    # train val test split
    train_data = data[:int(train_size * len(data))]
    val_data = data[int(train_size * len(data)):int(train_size * len(data)) + int(val_size * len(data))]
    test_data = data[int(train_size * len(data)) + int(test_size * len(data)):]
    
    # save files
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write("%s\n" % item)
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write("%s\n" % item)
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write("%s\n" % item)

def is_paper_valid(paper_id:str, fastlangid:LID=None)->bool:
    """Given a paper_id, checks if all following conditions hold: 
        - abstract and title should be both well-defined (both non-empty and not None)
        - paper should be in English. For brevity, we check if the abstract is in English.

    Args:
        paper_id (str): unique identifier inside the dataset.
        fastlangid (LID): language detection instance. Defaults to None.
        data (dict): dictionary mapping each paper id to textual files and information.

    Returns:
        bool: True if all the conditions hold, False otherwise
    """
    if not fastlangid:
        fastlangid = LID()
    if paper_id in data:
        paper = data[paper_id]
    else:
        return False
    # check all conditions
    return paper['abstract'] is not None and paper['title'] is not None and fastlangid.predict(paper["abstract"]) == 'en' and paper['abstract'] and paper['title']

def return_cited_paper_ids(paper_id:str, fastlangid:LID=None) -> Iterator[str]:
    """Given a paper_id, return the ids of all the valid cited papers.

    Args:
        paper_id (str): unique identifier inside the dataset.
        fastlangid (LID): language detection instance. Defaults to None.

    Yields:
        Iterator[str]: iterator containing all paper ids of the cited papers. 
    """
    # find the paper inside the dataset
    paper = data[paper_id]
    # iterate over all citations.
    for cited_paper in paper['references']:
        # check if cited paper satisfies the validity conditions
        if is_paper_valid(cited_paper, fastlangid = fastlangid):
            # return an Iterator of paper_ids
            yield cited_paper

def load_data(data_dir:str):
    """Given the path to the directory containing Scidocs metadata, merge them into a single dictionary.

    Args:
        data_dir (str): path of the directory containing the 3 metadata files

    Returns:
        dict: unique metadata dictionary. Maps each paper id to textual files and information.
    """
    path_mag_mesh = os.path.join(data_dir, 'paper_metadata_mag_mesh.json')
    path_recomm = os.path.join(data_dir, 'paper_metadata_recomm.json')
    path_cite = os.path.join(data_dir, 'paper_metadata_view_cite_read.json')

    with open(path_mag_mesh) as f:
        data_mag_mesh = json.load(f)

    with open(path_recomm) as f:
        data_recomm = json.load(f)

    with open(path_cite) as f:
        data_view_cite = json.load(f)

    data = dict(dict(data_mag_mesh, **data_recomm), **data_view_cite)
    return data

def main(data_dir:str, output_dir:str, checkpoint:bool, checkpoint_freq:int, train_size:float, val_size:float, test_size:float):
    """Create data.json file as per SPECTER specifications, starting from all Scidocs metadata files.

    Args:
        data_dir (str): directory containing Scidocs metadata files.
        output_dir (str): directory where the user wishes to save data.json
        checkpoint (bool, optional): parameter used to recover from crashes. Defaults to False.
        checkpoint_freq (int, optional): _description_. Defaults to 20_000.
    """
    # the dataset is a global variable that will be accessed by every function
    global data
    # create a unique metadata file with all Scidocs metadata.
    data = load_data(data_dir)
    # create the output directory, if it doesn't exist already
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    metadata_path = os.path.join(output_dir, 'metadata.json')
    # save metadata.json in the same folder as data.json
    with open(metadata_path, 'w') as f:
        json.dump(data, f)
    output_file = os.path.join(output_dir, 'data.json')

    # initialise a dictionary which will associate to each paper_id, its citation network as per data.json structure
    citation_network = {}

    if checkpoint:
        # set a counter for the number of citations. 
        # every time n_iter will reach 20k, we will get rid of useless data
        n_iter = 0

    # for each paper in the dataset
    for paper_id in tqdm(data.keys(), desc = "Scanning papers"):
        # Initialise language detection instance
        fastlangid = LID()
        # check if paper is valid
        if is_paper_valid(paper_id, fastlangid = fastlangid):
            # add a new key-value to the citation network
            #   - key is the paper_id
            #   - value is a dictionary with each cited paper_id. Can be empty.
            for cited_paper_id in return_cited_paper_ids(paper_id, fastlangid):
                # check if this is the first time that the 'father' paper is added to the dictionary
                # if not, update the corresponding dictionary value.
                if paper_id not in citation_network:
                    citation_network[paper_id] = {cited_paper_id : {"count" : 5}}
                else:
                    citation_network[paper_id].update({cited_paper_id : {"count" : 5}})
                # update the dictionary value with the citations of the citations.
                citation_network[paper_id].update({cited_cited_paper_id : {"count" : 1} for cited_cited_paper_id in return_cited_paper_ids(cited_paper_id, fastlangid)})
        
        if checkpoint:
            # you made an iteration
            n_iter += 1
            # check if `n_iter` is a multiple of `checkpoint_freq`. 
            # If so, update the checkpoint and reset the citation network
            if n_iter > 0 and n_iter % checkpoint_freq == 0:
                # check if this is the first checkpoint
                if not os.path.isfile(output_file):
                    with open(output_file, 'w') as f:
                        json.dump(citation_network, f)
                else:
                    # add the citation network to a pre existent checkpoint
                    with open(output_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    # Update the checkpoint with new citation network
                    checkpoint_data.update(citation_network)
                    # Write the updated checkpoint back to the file
                    with open(output_file, 'w') as f:
                        json.dump(checkpoint_data, f)

                    # clean checkpoint from memory
                    del checkpoint_data
                # empty citation network
                citation_network = {}

    # add last checkpoint
    if checkpoint:
        # add the citation network to a pre existent checkpoint
        with open(output_file, 'r') as f:
            checkpoint_data = json.load(f)
        # Update the checkpoint with new citation network
        checkpoint_data.update(citation_network)
        paper_keys = list(checkpoint_data.keys())
        # Write the updated checkpoint back to the file
        with open(output_file, 'w') as f:
            json.dump(checkpoint_data, f)
    # save the citation network.
    else:
        paper_keys = list(citation_network.keys())
        with open(output_file, 'w') as f:
            json.dump(citation_network, f)

    # create the train.txt, val.txt, test.txt file
    split_data(paper_keys, f'{output_dir}/train.txt', f'{output_dir}/val.txt', f'{output_dir}/test.txt', train_size, val_size, test_size)

    

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

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
    ap.add_argument('--data-dir', help='directory containing Scidocs metadata files', default = 'scidocs_metadata')
    ap.add_argument('--output-dir', help='directory where the user wishes to save data.json')
    ap.add_argument('--checkpoint', help='parameter used to recover from crashes',  default = False, type = boolean_string)
    ap.add_argument('--checkpoint-freq', help='how often do you want to dump the results', default=20000, type=int)
    ap.add_argument('--train-size', help='size of train split', type=restricted_float, default=0.7)
    ap.add_argument('--val-size', help='size of val split', type=restricted_float, default=0.15)
    
    args = ap.parse_args()
    train_size = args.train_size
    val_size = args.val_size
    test_size = 1 - train_size - val_size

    main(data_dir=args.data_dir, output_dir=args.output_dir, checkpoint=args.checkpoint, checkpoint_freq=args.checkpoint_freq,
    train_size = train_size, val_size = val_size, test_size = test_size)