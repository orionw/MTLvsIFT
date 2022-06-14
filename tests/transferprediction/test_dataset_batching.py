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
from transferprediction.utils import flatten


random.seed(42)


class DatasetBatching(unittest.TestCase):
    """
    NOTE: for all these tests, the instances aren't shuffled.  We would expect the
    sampling function to deal with the shuffling - batching should only send them
    into their batches.
    """

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
        cls.example_batches = [
            [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]],
            [["a2", "a3"], ["b2", "b3"], ["c2", "c3"]],
        ]
        cls.INSTANCES_EVEN = sum(
            len(flatten(item)) for item in cls.example_batches
        )
        cls.example_batches_uneven = [
            [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2"], ["c0", "c1"]],
            [["a4", "a5", "a6", "a7"], ["b3", "b4", "b5"], ["c2", "c3"]],
        ]
        cls.INSTANCES_UNEVEN = sum(
            len(flatten(item)) for item in cls.example_batches_uneven
        )

    def test_convering_batches(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        results = dataset.convert_heterogeneous_batches_to_homogenous(
            self.example_batches
        )
        expected_results = [
            ["a0", "a1", "a2", "a3"],
            ["b0", "b1", "b2", "b3"],
            ["c0", "c1", "c2", "c3"],
        ]
        assert (
            results == expected_results
        ), "converting batches to homogeneous did not work"

    def test_partitioning(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.instances = copy.deepcopy(self.example_batches)
        results = dataset.group_instances_by_partioning()
        expected_results = [
            "a0",
            "a1",
            "a2",
            "a3",
            "b0",
            "b1",
            "b2",
            "b3",
            "c0",
            "c1",
            "c2",
            "c3",
        ]
        assert (
            expected_results == results
        ), "failed to gather the correct instance list by partitioning: {}".format(
            results
        )

    def test_partitioning_uneven(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.instances = copy.deepcopy(self.example_batches_uneven)
        results = dataset.group_instances_by_partioning()
        expected_results = [
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "b0",
            "b1",
            "b2",
            "b3",
            "b4",
            "b5",
            "c0",
            "c1",
            "c2",
            "c3",
        ]
        assert (
            expected_results == results
        ), "failed to gather the correct instance list by partitioning: {}".format(
            results
        )

    def test_homogeneous(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        dataset.instances = copy.deepcopy(self.example_batches)
        results = dataset.group_instances_homogeneous()
        for index in range(self.INSTANCES_EVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            assert (
                len(set(first_letters)) == 1
            ), "did not group the same items in a batch"
            assert list(range(int(numbers[0]), int(numbers[1]) + 1)) == [
                int(item) for item in numbers
            ], "did not gather numbers sequentially"

        dataset.batch_size = 5  # more than instances available
        with self.assertRaises(Exception):
            results = dataset.group_instances_homogeneous()

    def test_homogeneous_uneven(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        dataset.instances = copy.deepcopy(self.example_batches_uneven)
        results = dataset.group_instances_homogeneous()
        for index in range(self.INSTANCES_UNEVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            assert (
                len(set(first_letters)) == 1
            ), "did not group the same items in a batch"
            assert list(range(int(numbers[0]), int(numbers[1]) + 1)) == [
                int(item) for item in numbers
            ], "did not gather numbers sequentially"

    def test_heterogeneous(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        dataset.instances = copy.deepcopy(self.example_batches)
        results = dataset.group_instances_heterogeneous()
        not_same_letters = False
        not_same_numbers = False
        for index in range(self.INSTANCES_EVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            if len(set(first_letters)) > 1:
                not_same_letters = True
            if list(range(int(numbers[0]), int(numbers[1]) + 1)) != [
                int(item) for item in numbers
            ]:
                not_same_numbers = True

        assert (
            not_same_letters
        ), "did not get a single instance of differing letters for heterogeneous"
        assert (
            not_same_numbers
        ), "did not get a single instance of out of sequences numbers for heterogeneous"

    def test_heterogeneous_uneven(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        dataset.instances = copy.deepcopy(self.example_batches_uneven)
        results = dataset.group_instances_heterogeneous()
        for index in range(self.INSTANCES_UNEVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            if len(set(first_letters)) > 1:
                not_same_letters = True
            if list(range(int(numbers[0]), int(numbers[1]) + 1)) != [
                int(item) for item in numbers
            ]:
                not_same_numbers = True

        assert (
            not_same_letters
        ), "did not get a single instance of differing letters for heterogeneous"
        assert (
            not_same_numbers
        ), "did not get a single instance of out of sequences numbers for heterogeneous"

        total = 0
        for sublist in self.example_batches_uneven:
            for subsublist in sublist:
                total += len(subsublist)
        assert (
            len(results) == total
        ), "did not use all of the data: {} vs total={}".format(
            len(results), total
        )

    def test_heterogeneous_forced(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        dataset.instances = copy.deepcopy(self.example_batches)
        results = dataset.group_instances_forced_heterogeneous()
        seen_chars = []
        for index in range(self.INSTANCES_EVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            assert (
                len(set(first_letters)) == 1
            ), "did not group the same items in a batch"
            assert list(range(int(numbers[0]), int(numbers[1]) + 1)) == [
                int(item) for item in numbers
            ], "did not gather numbers sequentially"
            seen_chars.append(first_letters[0])
        assert set(seen_chars) == set(
            dataset.dataset_list
        ), "did not see all tasks in the batch for forced heterogeneous"

    def test_heterogeneous_forced_uneven(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)
        dataset.dataset_list = ["a", "b", "c"]
        dataset.batch_size = 2
        # since the function expects them to already be in `batch_size` batches,
        # we can't give it the regular `self.example_batches_uneven` - it will put batches of three in
        uneven_batches_that_fit_batch_size = [
            [["a0", "a1", "a2", "a3"], ["b1", "b2"], ["c0", "c1"]],
            [["a4", "a5", "a6", "a7"], ["b4", "b5"], ["c2", "c3"]],
        ]
        dataset.instances = copy.deepcopy(uneven_batches_that_fit_batch_size)
        results = dataset.group_instances_forced_heterogeneous()

        seen_chars = []
        for index in range(self.INSTANCES_EVEN // 2):
            group = results[
                index * dataset.batch_size : (index + 1) * dataset.batch_size
            ]
            assert len(group) != 0, "empty group, error"
            first_letters = [item[0] for item in group]
            numbers = [item[-1] for item in group]
            assert (
                len(set(first_letters)) == 1
            ), "did not group the same items in a batch"
            assert list(range(int(numbers[0]), int(numbers[1]) + 1)) == [
                int(item) for item in numbers
            ], "did not gather numbers sequentially"
            seen_chars.append(first_letters[0])

        assert set(seen_chars) == set(
            dataset.dataset_list
        ), "did not see all tasks in the batch for forced heterogeneous"

        total = 0
        for sublist in uneven_batches_that_fit_batch_size:
            for subsublist in sublist:
                total += len(subsublist)
        assert (
            len(results) == total
        ), "did not use all of the data: {} vs total={}".format(
            len(results), total
        )
