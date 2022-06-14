"""Tests for transferprediction.multi_dataloader.py"""

import logging
import os
import unittest
from argparse import Namespace
import copy
import pytest
import random
import collections

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

from transferprediction.multi_dataloader import (
    MTLDataset,
    MTLRandomSampler,
)
from transferprediction.utils import flatten
from transferprediction.run_glue import load_and_cache_examples

# it thinks torch.tensor() is not callable
# pylint: disable=not-callable

random.seed(42)


class DatasetBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.OUTPUT_DIR = os.path.join("tests", "results")
        cls.TASK_NAMES = ["A", "B"]
        cls.args = Namespace(
            model_type="roberta",
            model_name_or_path="distilroberta-base",
            task_names=cls.TASK_NAMES,
            do_lower_case=True,
            max_seq_length=128,
            local_rank=-1,
            overwrite_cache=True,
            sampling_type="uniform",
            batch_type="heterogeneous",
        )
        training_datasets = []
        cls.batch_size = 2
        cls.num_instances = 10

    def test_no_duplicates_in_pairs(self):
        count = 0
        training_datasets = []
        for index in range(len(self.TASK_NAMES)):
            dataset = TensorDataset(
                torch.cat(
                    [
                        torch.tensor(i).unsqueeze(0)
                        for i in range(count, count + self.num_instances // 2)
                    ]
                )
            )
            training_datasets.append(dataset)
            count += self.num_instances // 2
        self.datasets = training_datasets
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, self.datasets, self.batch_size)
        sampler = MTLRandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        dataset_seen = []
        dataset_count = 0
        for (tensor_index, dataset_id) in dataset.arranged_instances:
            dataset_seen.append(tensor_index.item())
            dataset_count += 1

        assert set(dataset_seen) == set(
            list(range(self.num_instances))
        ), "did not use all unique items"
        assert (
            dataset_count == self.num_instances
        ), "got more or less than expected"

        seen = []
        total_count = 0
        for all_batch in dataloader:
            for task_batch, dataset_id_batch in all_batch:
                for item in task_batch.numpy().tolist():
                    assert len(item) == 1, "got too many in one instance"
                    seen.append(item[0])
                    total_count += 1

        assert set(seen) == set(
            list(range(self.num_instances))
        ), "did not use all unique items"
        assert (
            total_count == self.num_instances
        ), "got more or less than expected"

    def test_no_duplicates_in_singles(self):
        original = TensorDataset(
            torch.cat(
                [
                    torch.tensor(i).unsqueeze(0)
                    for i in range(self.num_instances)
                ]
            )
        )
        self.datasets = [original]
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(
            cur_args, copy.deepcopy(self.datasets), self.batch_size
        )
        dataset_seen = []
        dataset_count = 0

        for (tensor_index, dataset_id) in dataset.arranged_instances:
            dataset_seen.append(tensor_index.item())
            dataset_count += 1

        assert set(dataset_seen) == set(
            list(range(self.num_instances))
        ), "did not use all unique items"
        assert (
            dataset_count == self.num_instances
        ), "got more or less than expected"

        sampler = MTLRandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        seen = []
        total_count = 0
        for all_batch in dataloader:
            for task_batch, dataset_id_batch in all_batch:
                for item in task_batch.numpy().tolist():
                    assert len(item) == 1, "got too many in one instance"
                    seen.append(item[0])
                    total_count += 1

        assert set(seen) == set(
            list(range(self.num_instances))
        ), "did not use all unique items"
        assert (
            total_count == self.num_instances
        ), "got more or less than expected"

    def test_pair_and_single_multi(self):
        ## Create and test singles
        original = TensorDataset(
            torch.cat(
                [
                    torch.tensor(i).unsqueeze(0)
                    for i in range(self.num_instances)
                ]
            )
        )
        self.datasets = [original]
        cur_args = copy.deepcopy(self.args)
        dataset_single = MTLDataset(
            cur_args, copy.deepcopy(self.datasets), self.batch_size
        )
        sampler_single = MTLRandomSampler(dataset_single)
        dataloader_single = DataLoader(
            dataset_single,
            sampler=sampler_single,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        seen_single = []
        total_count_single = 0
        for epoch in range(5):
            for all_batch in dataloader_single:
                for task_batch, dataset_id_batch in all_batch:
                    for item in task_batch.numpy().tolist():
                        assert len(item) == 1, "got too many in one instance"
                        seen_single.append(item[0])
                        total_count_single += 1

        ## Create and test pairs
        count = 0
        training_datasets = []
        for index in range(len(self.TASK_NAMES)):
            dataset = TensorDataset(
                torch.cat(
                    [
                        torch.tensor(i).unsqueeze(0)
                        for i in range(count, count + self.num_instances // 2)
                    ]
                )
            )
            training_datasets.append(dataset)
            count += self.num_instances // 2

        self.datasets = training_datasets
        cur_args = copy.deepcopy(self.args)
        dataset_pairs = MTLDataset(cur_args, self.datasets, self.batch_size)
        sampler_pairs = MTLRandomSampler(dataset_pairs)
        dataloader_pairs = DataLoader(
            dataset_pairs,
            sampler=sampler_pairs,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        seen_pairs = []
        total_count_pairs = 0
        for epoch in range(5):
            for all_batch in dataloader_pairs:
                for task_batch, dataset_id_batch in all_batch:
                    for item in task_batch.numpy().tolist():
                        assert len(item) == 1, "got too many in one instance"
                        seen_pairs.append(item[0])
                        total_count_pairs += 1

        assert (
            total_count_pairs == total_count_single
        ), "did not train the same length"
        counter_pairs = collections.Counter(seen_pairs)
        counter_single = collections.Counter(seen_single)
        for i in range(self.num_instances):
            assert (
                counter_pairs[i] == counter_single[i]
            ), "did not get the same instances for num {}".format(i)

    def test_pairs_and_single_uneven(self):
        self.num_instances = 11
        ## Create and test singles
        original = TensorDataset(
            torch.cat(
                [
                    torch.tensor(i).unsqueeze(0)
                    for i in range(self.num_instances)
                ]
            )
        )
        self.datasets = [original]
        cur_args = copy.deepcopy(self.args)
        dataset_single = MTLDataset(
            cur_args, copy.deepcopy(self.datasets), self.batch_size
        )
        sampler_single = MTLRandomSampler(dataset_single)
        dataloader_single = DataLoader(
            dataset_single,
            sampler=sampler_single,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        seen_single = []
        total_count_single = 0
        for epoch in range(5):
            for all_batch in dataloader_single:
                for task_batch, dataset_id_batch in all_batch:
                    for item in task_batch.numpy().tolist():
                        assert len(item) == 1, "got too many in one instance"
                        seen_single.append(item[0])
                        total_count_single += 1

        ## Create and test pairs
        dataset_zero = TensorDataset(
            torch.cat(
                [
                    torch.tensor(i).unsqueeze(0)
                    for i in range(0, self.num_instances // 2)
                ]
            )
        )
        dataset_one = TensorDataset(
            torch.cat(
                [
                    torch.tensor(i).unsqueeze(0)
                    for i in range(self.num_instances // 2, self.num_instances)
                ]
            )
        )
        self.datasets = [dataset_zero, dataset_one]
        cur_args = copy.deepcopy(self.args)
        dataset_pairs = MTLDataset(cur_args, self.datasets, self.batch_size)
        sampler_pairs = MTLRandomSampler(dataset_pairs)
        dataloader_pairs = DataLoader(
            dataset_pairs,
            sampler=sampler_pairs,
            batch_size=self.batch_size,
            collate_fn=lambda x: [
                torch.utils.data.dataloader.default_collate(x)
            ],
        )

        seen_pairs = []
        total_count_pairs = 0
        for epoch in range(5):
            for all_batch in dataloader_pairs:
                for task_batch, dataset_id_batch in all_batch:
                    for item in task_batch.numpy().tolist():
                        assert len(item) == 1, "got too many in one instance"
                        seen_pairs.append(item[0])
                        total_count_pairs += 1

        assert (
            total_count_pairs == total_count_single
        ), "did not train the same length"
        counter_pairs = collections.Counter(seen_pairs)
        counter_single = collections.Counter(seen_single)
        pairs_counts = collections.Counter(counter_pairs.values())
        single_counts = collections.Counter(counter_single.values())
        assert (
            counter_single.keys() == counter_pairs.keys()
        ), "did not get the same number of runs for each instances (randomly ordered, of course)"
        assert (
            single_counts == pairs_counts
        ), "did not get the same number of counts"
