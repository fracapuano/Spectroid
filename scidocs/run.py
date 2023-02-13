import argparse
import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightning.classification import LinearSVC
from tqdm import tqdm



pd.set_option('display.max_columns', None)


def get_scidocs_metrics(data_paths,
                            classification_embeddings_path,
                            val_or_test='test',
                            n_jobs=-1):
        """This is the master wrapper that computes the SciDocs metrics given
        three embedding files (jsonl) and some optional parameters.
        Arguments:
            data_paths {scidocs.DataPaths} -- A DataPaths objects that points to 
                                            all of the SciDocs files
            classification_embeddings_path {str} -- Path to the embeddings jsonl 
                                                    for MAG and MeSH tasks
            n_jobs -- number of parallel jobs for classification related tasks (-1 to use all cpus)
            cuda_device -- cuda device for the recommender model (default is -1, meaning use CPU)
        Keyword Arguments:
            val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                                or the test set which is what's reported in SPECTER paper
                                (default: 'test')
            cuda_device {int} -- For the recomm pytorch model -> which cuda device to use(default: -1)
        Returns:
            scidocs_metrics {dict} -- SciDocs metrics for all tasks
        """
        assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"
        
        scidocs_metrics = {}
        scidocs_metrics.update(get_mag_mesh_metrics(data_paths, classification_embeddings_path, val_or_test=val_or_test, n_jobs=n_jobs))
        
        return scidocs_metrics

def get_mag_mesh_metrics(data_paths, embeddings_path=None, val_or_test='test', n_jobs=1):
        """Run MAG and MeSH tasks.
        Arguments:
            data_paths {scidocs.paths.DataPaths} -- A DataPaths objects that points to 
                                                    all of the SciDocs files
        Keyword Arguments:
            embeddings_path {str} -- Path to the embeddings jsonl (default: {None})
            val_or_test {str} -- Whether to return metrics on validation set (to tune hyperparams)
                                or the test set (what's reported in SPECTER paper)
        Returns:
            metrics {dict} -- F1 score for both tasks.
        """
        assert val_or_test in ('val', 'test'), "The val_or_test parameter must be one of 'val' or 'test'"
        
        print('Loading embeddings...')
        embeddings = load_embeddings_from_jsonl(embeddings_path)

        train_file = os.path.join(data_paths, 'train.csv')
        test_file = os.path.join(data_paths, 'test.csv')

        print('Running the classification task...')
        X, y = get_X_y_for_classification(embeddings, train_file, test_file)
        f1 = classify(X['train'], y['train'], X[val_or_test], y[val_or_test], n_jobs=n_jobs)
        
        return {'classification': {'f1': f1}}


def classify(X_train, y_train, X_test, y_test, n_jobs=1):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation. 
    
    Arguments:
        X_train, y_train -- training data
        X_test, y_test -- test data to evaluate on (can also be validation data)
    Returns: 
        F1 on X_test, y_test (out of 100), rounded to two decimal places
    """
    if not isinstance(y_train[0].item(), int):
        # Initialize the LabelEncoder
        le = LabelEncoder()

        # Encode the ground truth labels
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)
    estimator = LinearSVC(loss="squared_hinge", random_state=42)
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=estimator, cv=3, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    return np.round(100 * f1_score(y_test, preds, average='macro'), 2)


def get_X_y_for_classification(embeddings, train_path, test_path):
    """
    Given the directory with train/test/val files for mesh classification
        and embeddings, return data as X, y pair
        
    Arguments:
        embeddings: embedding dict
        mesh_dir: directory where the mesh ids/labels are stored
        dim: dimensionality of embeddings
    Returns:
        X, y: dictionaries of training X and training y
            with keys: 'train', 'val', 'test'
    """
    dim = len(next(iter(embeddings.values())))
    train = pd.read_csv(train_path, index_col = 0)
    test = pd.read_csv(test_path, index_col = 0)
    X = defaultdict(list)
    y = defaultdict(list)
    for dataset_name, dataset in zip(['train', 'test'], [train, test]):
        for s2id, class_label in dataset.iterrows():
            if str(s2id) not in embeddings:
                X[dataset_name].append(np.zeros(dim))
            else:
                X[dataset_name].append(embeddings[str(s2id)])
            y[dataset_name].append(class_label)
        X[dataset_name] = np.array(X[dataset_name])
        y[dataset_name] = np.array(y[dataset_name])
    return X, y

def load_embeddings_from_jsonl(embeddings_path):
    """Load embeddings from a jsonl file.
    The file must have one embedding per line in JSON format.
    It must have two keys per line: `paper_id` and `embedding`
    Arguments:
        embeddings_path -- path to the embeddings file
    Returns:
        embeddings -- a dictionary where each key is the paper id
                                and the value is a numpy array 
    """
    embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in tqdm(f, desc='reading embeddings from file...'):
            line_json = json.loads(line)
            embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '--classification-embeddings-path', dest='cls', help='path to classification related embeddings (mesh and mag)', default = 'testing_datasets/arxiv_data/output_specter.jsonl')
    parser.add_argument('--val_or_test', default='test', help='whether to evaluate scidocs on test data (what is reported in the specter paper) or validation data (to tune hyperparameters)')
    parser.add_argument('--n-jobs', default=12, help='number of parallel jobs for classification (related to mesh/mag metrics)', type=int)
    parser.add_argument('--data-path', help='path to the data directory where scidocs files reside. If None, it will default to the `data/` directory', default = "testing_datasets/arxiv_data/classification/")
    args = parser.parse_args()

    data_paths = os.path.join(os.getcwd(), args.data_path)

    scidocs_metrics = get_scidocs_metrics(
        data_paths,
        args.cls,
        val_or_test=args.val_or_test,
        n_jobs=args.n_jobs
    )

    flat_metrics = {}
    for k, v in scidocs_metrics.items():
        if not isinstance(v, dict):
            flat_metrics[k] = v
        else:
            for kk, vv in v.items():
                key = k + '-' + kk
                flat_metrics[key] = vv
    df = pd.DataFrame(list(flat_metrics.items())).T
    print(df)

if __name__ == '__main__':
    main()

    np.random.seed(1)


    