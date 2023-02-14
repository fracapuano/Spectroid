"run this script to create training pickled training files"
import logging
import os

import argparse
import json
import multiprocessing
import pathlib
import pickle
from time import time
from typing import Dict, Optional, Tuple, List, Any

import tqdm
from allennlp.common import Params
from allennlp.data import DatasetReader, TokenIndexer, Token, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter, BertBasicWordSplitter
from allennlp.training.util import datasets_from_params

from multiprocessing import Pool
import multiprocessing

from specter.data_utils import triplet_sampling_parallel

"""
START TRIPLET SAMPLING PARALLEL
"""
import numpy as np
import random
import operator
import math

np.random.seed(321)
random.seed(321)


def is_int(n):
    """ checks if a number is float. 2.0 is True while 0.3 is False"""
    return round(n) == n

_coviews = None
_margin_fraction = None
_paper_ids_set = None
_samples_per_query = None
_ratio_hard_negatives = None


# def _get_triplet(query, coviews, margin_fraction, paper_ids_set, samples_per_query, ratio_hard_negatives):
def _get_triplet(query):
    global _coviews
    global _margin_fraction
    global _paper_ids_set
    global _samples_per_query
    global _ratio_hard_negatives
    global _incitations

    if query not in _coviews:
        return
    # self.coviews[query] is a dictionary of format {paper_id: {count: 1, frac: 1}}
    candidates = [(k, v['count']) for k, v in _coviews[query].items()]
    candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    if len(candidates) > 1:
        coview_spread = candidates[0][1] - candidates[-1][1]
    else:
        coview_spread = 0
    margin = _margin_fraction * coview_spread  # minimum margin of coviews between good and bad sample

    # If distance is 1 increase margin to 1 otherwise any margin_fraction will pass
    if is_int(candidates[0][1]) and is_int(candidates[-1][1]) and coview_spread == 1:
        margin = np.ceil(margin)


    results = []

    # -------- hard triplets
    if len(candidates) > 2 and margin != 0:

        # find valid candidates by going through sorted
        # list and finding index of first sample with max coviews - margin
        for j in range(len(candidates)):
            if candidates[j][1] < (candidates[0][1] - margin):
                candidates_hard_neg = candidates[j:]
                break
            else:
                candidates_hard_neg = []

        neg_len = len(candidates_hard_neg)
        pos_len = len(candidates) - neg_len

        if neg_len > 0:

            # generate hard candidates
            n_hard_samples = math.ceil(_ratio_hard_negatives * _samples_per_query)
            # if there aren't enough candidates to generate enough unique samples
            # reduce the number of samples to make it possible for them to be unique
            if (pos_len * neg_len) < n_hard_samples:
                n_hard_samples = pos_len * neg_len

            for i in range(n_hard_samples):
                # find the negative sample first.
                neg = candidates_hard_neg[np.random.randint(len(candidates_hard_neg))]  # random neg sample from candidates

                candidates_pos = []
                # find the good sample. find valid candidates by going through sorted list
                # in reverse and finding index of first sample with bad sample + margin
                for j in range(len(candidates) - 1, -1, -1):
                    if candidates[j][1] > (neg[1] + margin):
                        candidates_pos = candidates[0:j + 1]
                        break

                if candidates_pos:
                    pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random pos sample from candidates

                    # append the good and bad samples with their coview number to output
                    results.append([query, pos, neg])

        n_easy_samples = _samples_per_query - len(results)

        # ---------- easy triplets

        # valid candidates are those with zeros
        candidates_zero = list(_paper_ids_set.difference([i[0] for i in candidates] + [query]))

        # find valid candidates for good sample by going through sorted list
        # in reverse and finding index of first sample with at least margin coviews
        # note: this is another way to write candidates_pos = [i for i in candidates if i[1] > margin]
        # but is much faster for large candidate arrays
        for j in range(len(candidates) - 1, -1, -1):
            if candidates[j][1] > margin + candidates[-1][1]:
                candidates_pos = candidates[0:j + 1]
                break
            else:
                candidates_pos = []

        # if there are no valid candidates for good rec, return None to trigger query resample
        if candidates and len(candidates_pos) > 0:
            easy_samples: List = []
            for i in range(n_easy_samples):
                pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random good sample from candidates
                neg = candidates_zero[np.random.randint(len(candidates_zero))]  # random zero
                easy_samples.append([query, pos, (neg, float("-inf"))])
            results.extend(easy_samples)

    return results

def _get_triplet_prob(query):
    global _coviews
    global _margin_fraction
    global _paper_ids_set
    global _samples_per_query
    global _ratio_hard_negatives
    global _incitations

    if query not in _coviews:
        return
    # self.coviews[query] is a dictionary of format {paper_id: {count: 1, frac: 1}}
    candidates = [(k, v['count']) for k, v in _coviews[query].items()]
    candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    if len(candidates) > 1:
        coview_spread = candidates[0][1] - candidates[-1][1]
    else:
        coview_spread = 0
    margin = _margin_fraction * coview_spread  # minimum margin of coviews between good and bad sample

    # If distance is 1 increase margin to 1 otherwise any margin_fraction will pass
    if is_int(candidates[0][1]) and is_int(candidates[-1][1]) and coview_spread == 1:
        margin = np.ceil(margin)


    results = []

    # -------- hard triplets
    if len(candidates) > 2 and margin != 0:

        # find valid candidates by going through sorted
        # list and finding index of first sample with max coviews - margin
        for j in range(len(candidates)):
            if candidates[j][1] < (candidates[0][1] - margin):
                candidates_hard_neg = candidates[j:]
                break
            else:
                candidates_hard_neg = []

        neg_len = len(candidates_hard_neg)
        pos_len = len(candidates) - neg_len

        if neg_len > 0:

            # generate hard candidates
            n_hard_samples = math.ceil(_ratio_hard_negatives * _samples_per_query)
            # if there aren't enough candidates to generate enough unique samples
            # reduce the number of samples to make it possible for them to be unique
            if (pos_len * neg_len) < n_hard_samples:
                n_hard_samples = pos_len * neg_len

            for i in range(n_hard_samples):
                # find the negative sample first.
                neg = candidates_hard_neg[np.random.randint(len(candidates_hard_neg))]  # random neg sample from candidates

                candidates_pos = []
                # find the good sample. find valid candidates by going through sorted list
                # in reverse and finding index of first sample with bad sample + margin
                for j in range(len(candidates) - 1, -1, -1):
                    if candidates[j][1] > (neg[1] + margin):
                        candidates_pos = candidates[0:j + 1]
                        break

                if candidates_pos:
                    incitations = [_incitations[candidate[0]] for candidate in candidates_pos]
                    incitations = [incitation / sum(incitations) for incitation in incitations]

                    pos = candidates_pos[np.random.choice(len(candidates_pos), p = incitations)]

                    # append the good and bad samples with their coview number to output
                    results.append([query, pos, neg])

        n_easy_samples = _samples_per_query - len(results)

        # ---------- easy triplets

        # valid candidates are those with zeros
        candidates_zero = list(_paper_ids_set.difference([i[0] for i in candidates] + [query]))

        # find valid candidates for good sample by going through sorted list
        # in reverse and finding index of first sample with at least margin coviews
        # note: this is another way to write candidates_pos = [i for i in candidates if i[1] > margin]
        # but is much faster for large candidate arrays
        for j in range(len(candidates) - 1, -1, -1):
            if candidates[j][1] > margin + candidates[-1][1]:
                candidates_pos = candidates[0:j + 1]
                break
            else:
                candidates_pos = []

        # if there are no valid candidates for good rec, return None to trigger query resample
        if candidates and len(candidates_pos) > 0:
            easy_samples: List = []
            for i in range(n_easy_samples):
                incitations = [_incitations[candidate[0]] for candidate in candidates_pos]
                incitations = [incitation / sum(incitations) for incitation in incitations]

                pos = candidates_pos[np.random.choice(len(candidates_pos), p = incitations)]  # random good sample from candidates
                neg = candidates_zero[np.random.randint(len(candidates_zero))]  # random zero
                easy_samples.append([query, pos, (neg, float("-inf"))])
            results.extend(easy_samples)

    return results

def generate_triplets(paper_ids, coviews, margin_fraction, samples_per_query, ratio_hard_negatives, query_ids, incitations, data_subset=None, n_jobs=1):
    global _coviews
    global _margin_fraction
    global _samples_per_query
    global _ratio_hard_negatives
    global _query_ids
    global _paper_ids_set
    global _incitations

    _coviews = coviews
    _margin_fraction = margin_fraction
    _samples_per_query = samples_per_query
    _ratio_hard_negatives = ratio_hard_negatives
    _query_ids = query_ids
    _paper_ids_set = set(paper_ids)
    _incitations = incitations
    logger.info(f'generating triplets with: samples_per_query:{_samples_per_query},'
                f'ratio_hard_negatives:{_ratio_hard_negatives}, margin_fraction:{_margin_fraction}')

    if n_jobs == 1:
        if _incitations:
            logger.info(f'embedding will be trained based on our improvement.')
            results = [_get_triplet_prob(query) for query in tqdm.tqdm(query_ids)]
        else:
            logger.info(f'specter-like training.')
            results = [_get_triplet(query) for query in tqdm.tqdm(query_ids)]
    elif n_jobs > 0:
        logger.info(f'running {n_jobs} parallel jobs to get triplets for {data_subset or "not-specified"} set')
        if _incitations:
            logger.info(f'embedding will be trained based on our improvement.')
        else:
            logger.info(f'specter-like training.')
        with Pool(n_jobs) as p:
            if _incitations:
                results = list(tqdm.tqdm(p.imap(_get_triplet_prob, query_ids), total=len(query_ids)))
            else:
                results = list(tqdm.tqdm(p.imap(_get_triplet, query_ids), total=len(query_ids)))
    else:
        raise RuntimeError(f"bad argument `n_jobs`={n_jobs}, `n_jobs` should be -1 or >0")
    for res in results:
        if res:
            for triplet in res:
                yield triplet

"""
END TRIPLET SAMPLING PARALLEL
"""



def init_logger(*, fn=None):

    # !!! here
    from importlib import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    logging_params = {
        'level': logging.INFO,
        'format': '%(asctime)s,%(msecs)d %(levelname)-3s [%(filename)s:%(lineno)d] %(message)s'
    }

    if fn is not None:
        logging_params['filename'] = fn

    logging.basicConfig(**logging_params)
    logging.info('reloading logger')


bert_params = {
    "do_lowercase": "true",
    "pretrained_model": "data/scivocab_scivocab_uncased/vocab.txt",
    "use_starting_offsets": "true"
}

NO_VENUE = '--no_venue--'


# ---------------
# instead of a class with its own constructor we define global variables
# and set their values using a `set_values` function
# this is not a clean design and has been done to support multiprocessing
# for more context, see: https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class/21345308#21345308

# global variables
_tokenizer = None
_token_indexers = None
_token_indexer_author_id = None
_token_indexer_author_position = None
_token_indexer_venue = None
_token_indexer_id = None
_max_sequence_length = None
_concat_title_abstract = None
_data_source = None
_included_text_fields = None

MAX_NUM_AUTHORS = 5
# ----------------

def _get_author_field(authors: List[str]):
    """
    Converts a list of author ids to their corresponding label and positions
    Args:
        authors: list of authors

    Returns:
        authors and their positions
    """
    global _token_indexer_author_id
    global _token_indexer_author_position
    global _tokenizer
    authors = ' '.join(authors)
    authors = _tokenizer.tokenize(authors)
    if len(authors) > MAX_NUM_AUTHORS:
        authors = authors[:MAX_NUM_AUTHORS - 1] + [authors[-1]]
    author_field = TextField(authors, token_indexers=_token_indexer_author_id)

    author_positions = ' '.join([f'{i:02}' for i in range(len(authors))])
    author_positions_tokens = _tokenizer.tokenize(author_positions)
    position_field = TextField(author_positions_tokens, token_indexers=_token_indexer_author_position)
    return author_field, position_field


def set_values(max_sequence_length: Optional[int] = -1,
               concat_title_abstract: Optional[bool] = None,
               data_source: Optional[str] = None,
               included_text_fields: Optional[str] = None
               ) -> None:
    # set global values
    # note: a class with __init__ would have been a better design
    # we have this structure for efficiency reasons: to support multiprocessing
    # as multiprocessing with class methods is slower
    global _tokenizer
    global _token_indexers
    global _token_indexer_author_id
    global _token_indexer_author_position
    global _token_indexer_venue
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    if _tokenizer is None:  # if not initialized, initialize the tokenizers and token indexers
        _tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter(do_lower_case=bert_params["do_lowercase"]))
        _token_indexers = {"bert": PretrainedBertIndexer.from_params(Params(bert_params))}
        _token_indexer_author_id = {"tokens": SingleIdTokenIndexer(namespace='author')}
        _token_indexer_author_position = {"tokens": SingleIdTokenIndexer(namespace='author_positions')}
        _token_indexer_venue = {"tokens": SingleIdTokenIndexer(namespace='venue')}
        _token_indexer_id = {"tokens": SingleIdTokenIndexer(namespace='id')}
    _max_sequence_length = max_sequence_length
    _concat_title_abstract = concat_title_abstract
    _data_source = data_source
    _included_text_fields = included_text_fields


def get_text_tokens(title_tokens, abstract_tokens, abstract_delimiter):
    """ concats title and abstract using a delimiter"""
    if title_tokens[-1] != Token('.'):
            title_tokens += [Token('.')]

    title_tokens = title_tokens + abstract_delimiter + abstract_tokens
    return title_tokens

def get_instance(paper):
    global _tokenizer
    global _token_indexers
    global _token_indexer_author_id
    global _token_indexer_author_position
    global _token_indexer_venue
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    included_text_fields = set(_included_text_fields.split())

    query_abstract_tokens = _tokenizer.tokenize(paper.get("query_abstract") or "")
    query_title_tokens = _tokenizer.tokenize(paper.get("query_title") or "")

    pos_abstract_tokens = _tokenizer.tokenize(paper.get("pos_abstract") or "")
    pos_title_tokens = _tokenizer.tokenize(paper.get("pos_title") or "")

    neg_abstract_tokens = _tokenizer.tokenize(paper.get("neg_abstract") or "")
    neg_title_tokens = _tokenizer.tokenize(paper.get("neg_title") or "")

    if _concat_title_abstract and 'abstract' in included_text_fields:
        abstract_delimiter = [Token('[SEP]')]
        query_title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
        pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
        neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
        query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []

    if 'authors' in included_text_fields and _max_sequence_length > 0:
        max_seq_len = _max_sequence_length - 15  # reserve max 15 tokens for author names
    else:
        max_seq_len = _max_sequence_length

    if _max_sequence_length > 0:
        query_abstract_tokens = query_abstract_tokens[:max_seq_len]
        query_title_tokens = query_title_tokens[:max_seq_len]
        pos_abstract_tokens = pos_abstract_tokens[:max_seq_len]
        pos_title_tokens = pos_title_tokens[:max_seq_len]
        neg_abstract_tokens = neg_abstract_tokens[:max_seq_len]
        neg_title_tokens = neg_title_tokens[:max_seq_len]

    if 'authors' in included_text_fields:
        source_author_text = ' '.join(paper.get("query_authors") or [])
        pos_author_text = ' '.join(paper.get("pos_authors") or [])
        neg_author_text = ' '.join(paper.get("neg_authors") or [])
        source_author_tokens = _tokenizer.tokenize(source_author_text)
        pos_author_tokens = _tokenizer.tokenize(pos_author_text)
        neg_author_tokens = _tokenizer.tokenize(neg_author_text)

        author_delimiter = [Token('[unused0]')]

        query_title_tokens = query_title_tokens + author_delimiter + source_author_tokens
        pos_title_tokens = pos_title_tokens + author_delimiter + pos_author_tokens
        neg_title_tokens = neg_title_tokens + author_delimiter + neg_author_tokens

    query_venue_tokens = _tokenizer.tokenize(paper.get('query_venue') or NO_VENUE)
    pos_venue_tokens = _tokenizer.tokenize(paper.get('pos_venue') or NO_VENUE)
    neg_venue_tokens = _tokenizer.tokenize(paper.get('neg_venue') or NO_VENUE)

    # pos_year_tokens = _tokenizer.tokenize(paper.get("pos_year"))
    # pos_body_tokens = _tokenizer.tokenize(paper.get("pos_body"))
    #
    # neg_year_tokens = _tokenizer.tokenize(paper.get("neg_year"))
    # neg_body_tokens = _tokenizer.tokenize(paper.get("neg_body"))

    fields = {
        "source_title": TextField(query_title_tokens, token_indexers=_token_indexers),
        "pos_title": TextField(pos_title_tokens, token_indexers=_token_indexers),
        "neg_title": TextField(neg_title_tokens, token_indexers=_token_indexers),
        "source_venue": TextField(query_venue_tokens, token_indexers=_token_indexer_venue),
        "pos_venue": TextField(pos_venue_tokens, token_indexers=_token_indexer_venue),
        "neg_venue": TextField(neg_venue_tokens, token_indexers=_token_indexer_venue),
        'source_paper_id': MetadataField(paper['query_paper_id']),
        "pos_paper_id": MetadataField(paper['pos_paper_id']),
        "neg_paper_id": MetadataField(paper['neg_paper_id']),
    }

    source_authors, source_author_positions = _get_author_field(paper.get("query_authors") or [])
    pos_authors, pos_author_positions = _get_author_field(paper.get("pos_authors") or [])
    neg_authors, neg_author_positions = _get_author_field(paper.get("neg_authors") or [])

    fields['source_authors'] = source_authors
    fields['source_author_positions'] = source_author_positions
    fields['pos_authors'] = pos_authors
    fields['pos_author_positions'] = pos_author_positions
    fields['neg_authors'] = neg_authors
    fields['neg_author_positions'] = neg_author_positions

    if not _concat_title_abstract:
        if query_abstract_tokens:
            fields["source_abstract"] = TextField(query_abstract_tokens, token_indexers=_token_indexers)
        if pos_abstract_tokens:
            fields["pos_abstract"] = TextField(pos_abstract_tokens, token_indexers=_token_indexers)
        if neg_abstract_tokens:
            fields["neg_abstract"] = TextField(neg_abstract_tokens, token_indexers=_token_indexers)

    if _data_source:
        fields["data_source"] = MetadataField(_data_source)

    return Instance(fields)

class TrainingInstanceGenerator:

    def __init__(self,
                 data,
                 metadata,
                 add_probabilities,
                 samples_per_query: int = 5,
                 margin_fraction: float = 0.5,
                 ratio_hard_negatives: float = 0.3,
                 data_source: str = None):
        self.samples_per_query = samples_per_query
        self.margin_fraction = margin_fraction
        self.ratio_hard_negatives = ratio_hard_negatives
        self.paper_feature_cache = {}
        self.metadata = metadata
        self.data_source = data_source

        self.data = data
        self.add_probabilities = add_probabilities
        # self.triplet_generator = TripletGenerator(
        #     paper_ids=list(metadata.keys()),
        #     coviews=data,
        #     margin_fraction=self.margin_fraction,
        #     samples_per_query=self.samples_per_query,
        #     ratio_hard_negatives=self.ratio_hard_negatives
        # )

    def _get_paper_features(self, paper: Optional[dict] = None) -> \
        Tuple[List[Token], List[Token], List[Token], int, List[Token]]:
        if paper:
            paper_id = paper.get('paper_id')
            if paper_id in self.paper_feature_cache:  # This function is being called by the same paper multiple times.
                return self.paper_feature_cache[paper_id]

            venue = paper.get('venue') or NO_VENUE
            year = paper.get('year') or 0
            body = paper.get('body')
            authors = paper.get('author-names')
            author_ids = paper.get('authors')
            references = paper.get('references')
            features = paper.get('abstract'), paper.get('title'), venue, year, body, author_ids, references
            self.paper_feature_cache[paper_id] = features
            return features
        else:
            return None, None, None, None, None, None, None

    def find_incitations(self):
        """
        It creates a dictionary which associates to each paper_id in self.data the associated number of incitations.
        """
        incitations_dict = {}
        # iterate over all the papers.
        for cited_papers in self.data.values():
            for cited_paper_id, cited_count in cited_papers.items():
                # we are only interested in direct citations, i.e. 'count' : 5
                if cited_count['count'] == 5:
                    # check if it is the first time that you find this paper_id
                    # if so, initialise the counter to 1
                    if cited_paper_id not in incitations_dict:
                        incitations_dict[cited_paper_id] = 1
                    # this is not the first time, i.e., increment the counter
                    else:
                        incitations_dict[cited_paper_id] += 1

        return incitations_dict

    def get_raw_instances(self, query_ids, subset_name=None, n_jobs=10, max_training_triplets:int=None):
        """
        Given query ids, it generates triplets and returns corresponding instances
        These are raw instances (i.e., untensorized objects )
        The output of this function is later used with DatasetConstructor.get_instance to convert raw fields to tensors

        Args:
            query_ids: list of query ids from which triplets are generated
            subset_name: Optional name to specify the subset (train, test, or val)

        Returns:
            list of instances (dictionaries)
            each instance is a dictionary containing fields corresponding to the `query`, `positive` and
                `negative` papers in the triplet: e.g., query_abstract, query_title, pos_title, neg_title, etc
        """
        logger.info('Generating triplets ...')
        count_success, count_fail = 0, 0
        # instances = []
        if self.add_probabilities:
            incitations_dict = self.find_incitations()
        else:
            incitations_dict = {}
        
        # counter to keep track of the number of triplets
        # used to cap the total number of training triplets 
        count_training_triplets = 0
        for triplet in generate_triplets(list(self.metadata.keys()), self.data,
                                                            self.margin_fraction, self.samples_per_query,
                                                            self.ratio_hard_negatives, query_ids, incitations_dict,
                                                            data_subset=subset_name, n_jobs=n_jobs):
            
            if max_training_triplets and count_training_triplets == max_training_triplets:
                break
            try:
                query_paper = self.metadata[triplet[0]]
                pos_paper = self.metadata[triplet[1][0]]
                neg_paper = self.metadata[triplet[2][0]]
                count_success += 1

                # check if all papers have title and abstract (all must have title)
                failed = False
                for paper in (query_paper, pos_paper, neg_paper):
                    if not paper['title'] or (not paper['title'] and not paper['abstract']):
                        failed = True
                        break
                if failed:
                    count_fail += 1
                    continue

                query_abstract, query_title, query_venue, query_year, query_body, query_authors, query_refs = \
                    self._get_paper_features(query_paper)
                pos_abstract, pos_title, pos_venue, pos_year, pos_body, pos_authors, pos_refs = self._get_paper_features(pos_paper)
                neg_abstract, neg_title, neg_venue, neg_year, neg_body, neg_authors, neg_refs = self._get_paper_features(neg_paper)

                count_training_triplets += 1

                instance = {
                    "query_abstract": query_abstract,
                    "query_title": query_title,
                    "query_venue": query_venue,
                    "query_year": query_year,
                    "query_body": query_body,
                    "query_authors": query_authors,
                    "query_paper_id": query_paper["paper_id"],
                    "pos_abstract": pos_abstract,
                    "pos_title": pos_title,
                    "pos_venue": pos_venue,
                    "pos_year": pos_year,
                    "pos_body": pos_body,
                    "pos_authors": pos_authors,
                    "pos_paper_id": pos_paper["paper_id"],
                    "neg_abstract": neg_abstract,
                    "neg_title": neg_title,
                    "neg_venue": neg_venue,
                    "neg_year": neg_year,
                    "neg_body": neg_body,
                    "neg_authors": neg_authors,
                    "neg_paper_id": neg_paper["paper_id"],
                    "data_source": self.data_source
                }
                yield instance
            except KeyError:
                # if there is no title and abstract skip this triplet
                count_fail += 1
                pass
        logger.info(f"done getting triplets, success rate:{(count_success*100/(count_success+count_fail+0.001)):.2f}%,"
                     f"total: {count_success+count_fail}")


def get_instances(data, query_ids_file, metadata, add_probabilities = False, data_source=None, n_jobs=1, n_jobs_raw=12,
                  ratio_hard_negatives=0.3, margin_fraction=0.5, samples_per_query=5,
                  concat_title_abstract=False, included_text_fields='title abstract', max_training_triplets:int=None):
    """
    Gets allennlp instances from the data file
    Args:
        data: the data file (e.g., coviews, or cocites)
        query_ids_file: train.csv file (one query paper id per line)
        metadata: a json file containing mapping between paper ids and paper dictionaries
        add_probabilities: our improvement 
        data_source: the source of the data (e.g., is it coviews, cite1hop, copdf?)
        n_jobs: number of jobs to process allennlp instance conversion
        n_jobs_raw: number of jobs to generate raw triplets
        ratio_hard_negatives: how many hard nagatives
        margin_fraction: margin fraction param of triplet generation
        samples_per_query: how many smaples for each query paper

    Returns:
        List[Instance]
    """

    if n_jobs == 0:
        raise RuntimeError(f"argument `n_jobs`={n_jobs} is invalid, should be >0")

    generator = TrainingInstanceGenerator(data=data, metadata=metadata, data_source=data_source,
                                          margin_fraction=margin_fraction, ratio_hard_negatives=ratio_hard_negatives,
                                          samples_per_query=samples_per_query, add_probabilities = add_probabilities)

    set_values(max_sequence_length=512,
               concat_title_abstract=concat_title_abstract,
               data_source=data_source,
               included_text_fields=included_text_fields)

    query_ids = [line.strip() for line in open(query_ids_file)]

    instances_raw = [e for e in generator.get_raw_instances(
        query_ids, subset_name=query_ids_file.split('/')[-1][:-4], n_jobs=n_jobs_raw, max_training_triplets=max_training_triplets)]

    if n_jobs == 1:
        logger.info(f'converting raw instances to allennlp instances:')
        for e in tqdm.tqdm(instances_raw):
            yield get_instance(e)

    else:
        logger.info(f'converting raw instances to allennlp instances ({n_jobs} parallel jobs)')
        with Pool(n_jobs) as p:
            instances = list(tqdm.tqdm(p.imap(
                get_instance, instances_raw)))

        # multiprocessing does not work as generator, needs to generate everything
        # see: https://stackoverflow.com/questions/5318936/python-multiprocessing-pool-lazy-iteration
        return instances


def main(data_files, train_ids, val_ids, test_ids, metadata_file, outdir, n_jobs=1, njobs_raw=1,
         margin_fraction=0.5, ratio_hard_negatives=0.3, samples_per_query=5, comment='', bert_vocab='',
         concat_title_abstract=False, included_text_fields='title abstract', add_probabilities=False,
         max_training_triplets:int=None):
    """
    Generates instances from a list of datafiles and stores them as a stream of objects
    Args:
        data_files: list of files indicating cooccurrence counts
        train_ids: list of training paper ids corresponding to each data file
        val_ids: list of validation paper ids
        test_ids: list of test paper ids
        metadata_file: path to the metadata file (should cover all data files)
        outdir: path to an output directory
        add_probabilities: our improvement 
        n_jobs: number of parallel jobs for converting instances
            (in practice we did not find parallelization to help with this)
        njobs_raw: number of parallel jobs for generating triplets
            (in practice we found njobs_raw between 10 to 15 to perform fastests)
        margin_fraction: parameter for triplet generation
        ratio_hard_negatives: how many of triplets are hard negatives
        samples_per_query: how many samples per query paper
        comment: custom comment to add to the file name when saved


    Returns:
        Nothing
            Creates files corresponding to each data file
    """
    global bert_params
    bert_params['pretrained_model'] = bert_vocab

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(metadata_file) as f_in:
        logger.info(f'loading metadata: {metadata_file}')
        metadata = json.load(f_in)

    for data_file, train_set, val_set, test_set in zip(data_files, train_ids, val_ids, test_ids):
        logger.info(f'loading data file: {data_file}')
        with open(data_file) as f_in:
            data = json.load(f_in)
        data_source = data_file.replace('/', '\\').split('\\')[-1][:-5]  # e.g., coviews_v2012
        if comment:
            data_source += f'-{comment}'

        metrics = {}
        for ds_name, ds in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
            logger.info(f'getting instances for `{data_source}` and `{ds_name}` set')
            outfile = f'{outdir}/{data_source}-{ds_name}.p'
            logger.info(f'writing output {outfile}')
            with open(outfile, 'wb') as f_in:
                pickler = pickle.Pickler(f_in)
                # pickler.fast = True
                idx = 0
                for instance in get_instances(data=data,
                                              query_ids_file=ds,
                                              metadata=metadata,
                                              data_source=data_source,
                                              n_jobs=n_jobs, n_jobs_raw=njobs_raw,
                                              margin_fraction=margin_fraction,
                                              ratio_hard_negatives=ratio_hard_negatives,
                                              samples_per_query=samples_per_query,
                                              concat_title_abstract=concat_title_abstract,
                                              included_text_fields=included_text_fields,
                                              add_probabilities=add_probabilities,
                                              max_training_triplets=max_training_triplets if ds_name == 'train' else None
                                              ):
                    pickler.dump(instance)
                    idx += 1
                    # to prevent from memory blow
                    if idx % 2000 == 0:
                        pickler.clear_memo()
            metrics[ds_name] = idx
        with open(f'{outdir}/{data_source}-metrics.json', 'w') as f_out2:
            json.dump(metrics, f_out2, indent=2)

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', help='path to a directory containing `data.json`, `train.csv`, `dev.csv` and `test.csv` files', default = 'project_data')
    ap.add_argument('--metadata', help='path to the metadata file',  default = 'project_data/metadata.json')
    ap.add_argument('--outdir', help='output directory to files', default = 'project_data/preprocessed_improvement')
    ap.add_argument('--njobs', help='number of parallel jobs for instance conversion', default=1, type=int)
    ap.add_argument('--njobs_raw', help='number of parallel jobs for triplet generation', default=1, type=int)
    ap.add_argument('--ratio_hard_negatives', default=0.3, type=float)
    ap.add_argument('--samples_per_query', default=5, type=int)
    ap.add_argument('--margin_fraction', default=0.5, type=float)
    ap.add_argument('--comment', default='', type=str)
    ap.add_argument('--data_files', help='space delimted list of data files to override default', default=None)
    ap.add_argument('--bert_vocab', help='path to bert vocab', default='data/scibert_scivocab_uncased/vocab.txt')
    ap.add_argument('--concat-title-abstract', action='store_true', default=False)
    ap.add_argument('--included-text-fields', default='title abstract', help='space delimieted list of fields to include in the main text field'
                                                                             'possible values: `title`, `abstract`, `authors`')
    ap.add_argument('--add-probabilities', default = True, type = boolean_string, help = 'whether you want to use our improvement or pure SPECTER')    
    ap.add_argument('--max-training-triplets', default = 150_000, type = int, help = 'maximum number of training triplets')                                                                     
    args = ap.parse_args()

    data_file = os.path.join(args.data_dir, 'data.json')
    train_ids = os.path.join(args.data_dir, 'train.txt')
    val_ids = os.path.join(args.data_dir, 'val.txt')
    test_ids = os.path.join(args.data_dir, 'test.txt')

    if args.metadata:
        metadata_file = args.metadata

    init_logger()
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

    main([data_file], [train_ids], [val_ids], [test_ids], metadata_file, args.outdir, args.njobs, args.njobs_raw,
         margin_fraction=args.margin_fraction, ratio_hard_negatives=args.ratio_hard_negatives,
         samples_per_query=args.samples_per_query, comment=args.comment, bert_vocab=args.bert_vocab,
         concat_title_abstract=args.concat_title_abstract, included_text_fields=args.included_text_fields,
         add_probabilities=args.add_probabilities, max_training_triplets=args.max_training_triplets
         )
