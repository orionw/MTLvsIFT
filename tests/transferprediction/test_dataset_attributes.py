"""Tests for transferprediction.multi_dataloader.py"""

import logging
import os
import unittest
from argparse import Namespace
import copy
import pytest
import random

import numpy as np
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

from transferprediction.multi_dataloader import MTLDataset, flatten
from transferprediction.run_glue import load_and_cache_examples

random.seed(42)


class DatasetBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OUTPUT_DIR = os.path.join("tests", "results")
        cls.TASK_NAMES = ["CoLA", "SST-2", "QQP", "STS-B"]
        cls.data_paths = [
            os.path.join("tests/fixtures/dataset", task)
            for task in cls.TASK_NAMES
        ]
        cls.tokenizer = RobertaTokenizer.from_pretrained(
            "distilroberta-base",
            do_lower_case=True,
            cache_dir=None,
        )
        data_dirs = {
            task_name.lower(): data_path
            for task_name, data_path in zip(cls.TASK_NAMES, cls.data_paths)
        }
        cls.args = Namespace(
            data_dirs=data_dirs,
            model_type="roberta",
            model_name_or_path="distilroberta-base",
            task_names=cls.TASK_NAMES,
            do_lower_case=True,
            max_seq_length=128,
            local_rank=-1,
            overwrite_cache=True,
            sampling_type="uniform",
            batch_type="partition",
        )
        training_datasets = []
        for task_name, data_path in zip(cls.TASK_NAMES, cls.data_paths):
            training_datasets.append(
                load_and_cache_examples(
                    cls.args,
                    task_name.lower(),
                    data_path,
                    cls.tokenizer,
                    evaluate=False,
                )
            )
        cls.datasets = training_datasets
        cls.batch_size = 64
        cls.total_size = 1000  # len of the original data in the fixtures folder

    def test_error_wrong_batch_type(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.batch_type = "unknown"
        with self.assertRaises(NotImplementedError):
            dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)

    def test_error_wrong_sampling_type(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.batch_type = "unknown"
        with self.assertRaises(NotImplementedError):
            dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)

    def test_properties_saved(self):
        cur_datasets = copy.deepcopy(self.datasets)
        dataset_full = MTLDataset(self.args, cur_datasets, self.batch_size)
        assert dataset_full.args == self.args, "did not save args"
        assert (
            dataset_full.batch_size == self.batch_size
        ), "did not save batch size"
        assert len(dataset_full.all_instances) == len(
            self.datasets
        ), "did not gather the all_instances correctly"
        assert (
            dataset_full.all_instances[0][0][-1] == 0
        ), "did not save dataset index in all_instances"
