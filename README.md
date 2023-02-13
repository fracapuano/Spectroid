![plot](https://i.ibb.co/3TC1WmG/specter-logo-cropped.png)

# SPECTER: Document-level Representation Learning using Citation-informed Transformers

> **NOTE**: Similarly to the [original repo](https://github.com/allenai/specter), this repo does not work on Windows.

## 0- Download repo

```python
conda create --name specter python=3.7 setuptools  

conda activate specter  

# if you don't have gpus, remove cudatoolkit argument
conda install pytorch cudatoolkit=10.1 -c pytorch   

pip install -r requirements.txt  

python setup.py install

pip install overrides==3.1.0

bash/download.sh
```
After running these commands, the environment will be set up and additional files will be downloaded. In particular, in the folder `pretrained_models` one could find the two models we trained: SPECTER and SPECTROID. These models are fully functional and can be used to embed your datasets as per point 4 of the documentation, passing one of the two models' filename as parameter of `--model`.

To have a full grasp of our work, please stick to the instructions here below.

## 1- Creation of the Scidocs dataset
To train an embedder, we created a new dataset starting from the metadata available on the [Scidocs repository](https://github.com/allenai/scidocs). Specifically, the dataset we created is made of:

* `data.json` containing the document ids and their relationship.  
* `metadata.json` containing mapping of document ids to textual fiels (e.g., `title`, `abstract`)
* `train.txt`, `val.txt`, `test.txt` containing document ids corresponding to train/val/test sets (one doc id per line).

The `data.json` file has the following structure (a nested dict):  
```python
{"docid1" : {  "docid11": {"count": 1}, 
               "docid12": {"count": 5},
               "docid13": {"count": 1}, ....
            }
"docid2":   {  "docid21": {"count": 1}, ....
            }
....}
```

Where `docids` are Scidocs paper ids and `count` is a measure of importance of the relationship between two documents. We use the same notation used in Specter, namely `count=5` means direct citation while `count=1` refers to a citation of a citation. 
The dataset is already available in the folder `project_data`.

For full reproducibility, it is also possible to create the same dataset from scratch starting from the Scidocs metadata files. Stemming from the files inside the folder `scidocs_metadata`, simply run:

```python
python scripts/create_scidocs.py --data-dir scidocs_metadata --output-dir project_data
```

## 2- Create training triplets
The `create_training_files.py` script processes this structure with a triplet sampler. The triplets can be formed either according to what described in the paper, or according to our improvement. Papers with `count=5` are considered positive candidates, papers with `count=1` considered hard negatives and other papers that are not cited are easy negatives. The number of hard negatives can be controlled by setting `--ratio_hard_negatives` argument in the script. The triplets can be formed as follows:
  
```python
python specter/data_utils/create_training_files.py --data-dir project_data --metadata project_data/metadata.json --outdir project_data/preprocessed_improvement/ --max-training-triplets 150000 --add-probabilities True 
```

`add-probabilities` is the parameter to set to choose between our improvement and Specter-like triplets.  It is advisable changing the `outdir` according to this parameter.

Due to limited resources, we also an optional parameter to set the maximum number of triplets.

Once the triplets are trained, the embedder can be trained as follows:

```python
./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet -s model-output-improvement/ --num-epochs 2 --batch-size 4 --train-path project_data/preprocessed-improvement/data-train.p --dev-path project_data/preprocessed-improvement/data-val.p --num-train-instances 150000 --cuda-device 0

```

In this example: The model's checkpoint and logs will be stored in `model-output-improvement/ `.  
Note that you need to set the correct `--num-train-instances` for your dataset. This number is either the maximum number of training triplets previously specified (if specified), or could be found in `data-metrics.json` file output from the preprocessing step. 

## 3- Creation of the embedding datasets
Specter requires two main files as input to embed the document. A text file with ids of the documents you want to embed and a json metadata file consisting of the title and abstract information. We embedded two different documents:
1. all the Scidocs papers for Mag and Mesh classification task
2. subset of 50000 arXiv papers.

The two datasets are already available in the folder `testing_datasets`. This folder contains two subfolders, one for each dataset. Each subfolder contains a text file with the ids of the document, the corresponding json metadata file and the train/val/test split of the ids (in .csv file).

For full reproducibility, one can create the `testing_datasets/arxiv_data` folder from scratch, using the Kaggle interface. Please follow this [guide](https://technowhisp.com/kaggle-api-python-documentation/) to correctly set-up the Kaggle Python API first.
```python
python scripts/create_arxiv.py --output-dir 'kaggle_dat/'
```

## 4- Embedding of the datasets
To use the previously trained model to embed your data, run the following:

```python
python scripts/embed.py --ids testing_datasets/arxiv_data/arxiv.ids --metadata testing_datasets/arxiv_data/metadata.json --model model-output-improvement/model.tar.gz --output-file testing_datasets/arxiv_data/output_improvement.jsonl --vocab-dir data/vocab/ --batch-size 64 --cuda-device 0

```
The model will run inference on the provided input and writes the output to `--output-file` directory (in the above example `output_improvement.jsonl` ).  
This is a jsonlines file where each line is a key, value pair consisting the id of the embedded document and its specter representation.

To use the Mag\Mesh Scidocs papers, it is enough to change `ids` and `metadata` (and possibly `output-file`) accordingly.

## 5- classification
The performance of our embedder can be evaluated as follows:

```python
python scidocs/run.py 
--cls testing_datasets/arxiv_data/output_improvement.jsonl 
--data-path testing_datasets/arxiv_data/classification/ 
--val_or_test test 
--n-jobs 12
```
