"""Tests for transferprediction.multi_dataloader.py"""

import logging
import os
import unittest
from argparse import Namespace
import copy
import pytest
import random
import math

import numpy as np
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

from transferprediction.multi_dataloader import MTLDataset, flatten
from transferprediction.run_glue import load_and_cache_examples

random.seed(42)


class DatasetSampling(unittest.TestCase):
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

    def test_size_batching(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "size"
        dataset_full = MTLDataset(cur_args, cur_datasets, self.batch_size)
        samples = dataset_full.arrange_instances_by_size()
        instances_per_batch = 16  # since they're all the same we can check only the first, aka we're using len(samples[0])
        assert len(samples) * len(samples[0]) == sum(
            [len(dataset) // instances_per_batch for dataset in self.datasets]
        ), "did not get the right amount of batches"

    def test_size_batching_uneven(self):
        lengths = [1000, 100]
        batch_size = 1
        cur_datasets = copy.deepcopy(self.datasets)
        uneven_datasets = [
            TensorDataset(*cur_datasets[i][0 : lengths[i]])
            for i in range(len(lengths))
        ]
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "size"
        dataset_full = MTLDataset(cur_args, uneven_datasets, self.batch_size)
        samples = dataset_full.arrange_instances_by_size()
        assert (
            len(samples[0]) == 2
        ), "did not gather the right number of dataset per batch"
        assert (
            len(set([len(sample[0]) for sample in samples])) == 1
        ), "had inconsistent batch sizes in the first dataset"
        assert (
            len(set([len(sample[1]) for sample in samples])) == 1
        ), "had inconsistent batch sizes in the second dataset"
        assert (
            round(len(samples[0][0]) / len(samples[0][1])) == 10.0
        ), "did not gather the right proportion in each batch"
        assert (
            sum(dataset_full.instances_per_batch) == self.batch_size
        ), "did not get the right instances per batch"
        # NOTE: because it rounds the batch sizes, it ends up being slightly sub-optimal in packing batches
        assert (
            len(samples)
            * (len(samples[0][0]) + len(samples[0][1]))
            / sum(lengths)
            > 0.9
        ), "did not gather enough samples"

    def test_uniform_batching(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "uniform"
        dataset_full = MTLDataset(cur_args, cur_datasets, self.batch_size)
        samples = dataset_full.arrange_instances_uniform()
        instances_per_batch = 16  # since they're all the same we can check only the first, aka we're using len(samples[0])
        assert len(samples) * len(samples[0]) == sum(
            [len(dataset) // instances_per_batch for dataset in self.datasets]
        ), "did not get the right amount of batches"

    def test_uniform_batching_uneven(self):
        lengths = [1000, 100]
        batch_size = 1
        cur_datasets = copy.deepcopy(self.datasets)
        uneven_datasets = [
            TensorDataset(*cur_datasets[i][0 : lengths[i]])
            for i in range(len(lengths))
        ]
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "uniform"
        dataset_full = MTLDataset(cur_args, uneven_datasets, self.batch_size)
        samples = dataset_full.arrange_instances_uniform()
        assert (
            len(samples[0]) == 2
        ), "did not gather the right number of dataset per batch"
        assert (
            len(set([len(sample[0]) for sample in samples])) == 1
        ), "had inconsistent batch sizes in the first dataset"
        assert (
            len(set([len(sample[1]) for sample in samples])) == 1
        ), "had inconsistent batch sizes in the second dataset"
        assert (
            round(len(samples[0][0]) / len(samples[0][1])) == 1.0
        ), "did not gather the right proportion in each batch"
        assert (
            sum(dataset_full.instances_per_batch) == self.batch_size
        ), "did not get the right instances per batch"
        # because it rounds the batch sizes, it ends up being slightly sub-optimal in packing batches, see above note
        # 192 is due to (min(lengths) which is 100 for both datasets: 200 // (batch_size // 2) *  batch_size
        assert (
            len(samples) * (len(samples[0][0]) + len(samples[0][1])) == 192
        ), "did not gather enough samples"

    def test_dynamic_sampling_ratio(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "dynamic"
        dataset_full = MTLDataset(
            cur_args,
            cur_datasets,
            self.batch_size,
            single_task_scores=[0.3, 0.75, 1.0, 1.0],
        )
        evaluation_metrics = {
            "CoLA": {"mcc": 0.1},
            "SST-2": {"acc": 0.5},
            "QQP": {"acc_and_f1": 0.5},
            "STS-B": {"spearmanr": 1.0},
        }
        expected_probs = [0.23697465, 0.2491246, 0.31988232, 0.19401843]
        sampling_probs = dataset_full.get_dynamic_sampling_distribution(
            evaluation_metrics
        )
        assert len(sampling_probs) == len(
            expected_probs
        ), "did not return same length vectors"
        for index in range(len(sampling_probs)):
            assert np.isclose(
                sampling_probs[index], expected_probs[index]
            ), "did not match on task {}".format(index)

        evaluation_metrics = {
            "CoLA": {"mcc": 0.3},
            "SST-2": {"acc": 0.75},
            "QQP": {"acc_and_f1": 0.0},
            "STS-B": {"spearmanr": 1.0},
        }
        expected_probs = [0.1748777, 0.1748777, 0.47536689, 0.1748777]
        sampling_probs = dataset_full.get_dynamic_sampling_distribution(
            evaluation_metrics
        )
        assert len(sampling_probs) == len(
            expected_probs
        ), "did not return same length vectors"
        for index in range(len(sampling_probs)):
            assert np.isclose(
                sampling_probs[index], expected_probs[index]
            ), "did not match on task {}".format(index)

    def test_dynamic_batching(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "dynamic"
        dataset_full = MTLDataset(
            cur_args,
            cur_datasets,
            self.batch_size,
            single_task_scores=[0.3, 0.75, 1.0, 1.0],
        )
        evaluation_metrics = {
            "CoLA": {"mcc": 0.1},
            "SST-2": {"acc": 0.5},
            "QQP": {"acc_and_f1": 0.5},
            "STS-B": {"spearmanr": 1.0},
        }
        samples = dataset_full.arrange_instances_dynamically(evaluation_metrics)
        max_samples_per_batch = max(dataset_full.instances_per_batch)
        assert (
            len(samples) * max_samples_per_batch
            >= max([len(dataset) for dataset in self.datasets])
            - max_samples_per_batch
        ), "did not get the right amount of batches"

    def test_dynamic_batching_uneven(self):
        lengths = [1000, 100]
        batch_size = 1
        cur_datasets = copy.deepcopy(self.datasets)
        uneven_datasets = [
            TensorDataset(*cur_datasets[i][0 : lengths[i]])
            for i in range(len(lengths))
        ]
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "dynamic"
        cur_args.task_names = ["CoLA", "SST-2"]
        dataset_full = MTLDataset(
            cur_args,
            uneven_datasets,
            self.batch_size,
            single_task_scores=[0.3, 0.75],
        )
        evaluation_metrics = {"CoLA": {"mcc": 0.1}, "SST-2": {"acc": 0.5}}
        samples = dataset_full.arrange_instances_dynamically(evaluation_metrics)
        assert dataset_full.instances_per_batch == [
            31,
            33,
        ], "had inconsistent batch sizes "
        assert len(set([len(sample[0]) for sample in samples])) in [
            1,
            2,
        ], (  # potential for end batch size to be smaller
            "had inconsistent batch sizes in the first dataset"
        )
        assert len(set([len(sample[1]) for sample in samples])) in [
            1,
            2,
        ], "had inconsistent batch sizes in the second dataset"
        assert (
            round(len(samples[0][0]) / len(samples[0][1])) == 1.0
        ), "did not gather the right proportion in each batch"
        # find the smallest dataset (index 1) and make sure we have the right number of batches
        assert lengths[1] // len(samples[0][1]) == len(
            samples
        ), "did not gather enough batches"

    def test_instance_batching_proportion_count(self):
        cur_datasets = copy.deepcopy(self.datasets)
        cur_args = copy.deepcopy(self.args)
        cur_args.sampling_type = "uniform"
        dataset = MTLDataset(cur_args, cur_datasets, self.batch_size)

        dataset.batch_size = 64
        results = dataset.get_instances_numbers_from_proportions(
            [0.25, 0.25, 0.25, 0.25]
        )
        assert results == [
            16,
            16,
            16,
            16,
        ], "did not split evenly, got: {}".format(results)

        dataset.batch_size = 16
        with self.assertRaises(Exception):
            results = dataset.get_instances_numbers_from_proportions(
                [0.25, 0.25, 0.25, 0.5]
            )

        dataset.batch_size = 2
        with self.assertRaises(Exception):
            results = dataset.get_instances_numbers_from_proportions(
                [0.25, 0.25, 0.5]
            )

        dataset.batch_size = 20
        results = dataset.get_instances_numbers_from_proportions(
            [0.4, 0.3, 0.2, 0.1]
        )
        assert results == [8, 6, 4, 2], "did not split cleanly, got: {}".format(
            results
        )

        dataset.batch_size = 16
        results = dataset.get_instances_numbers_from_proportions(
            [0.4, 0.3, 0.2, 0.1]
        )
        assert results == [6, 5, 3, 2], "did not split rough, got: {}".format(
            results
        )

        dataset.batch_size = 4
        results = dataset.get_instances_numbers_from_proportions(
            [0.4, 0.3, 0.2, 0.1]
        )
        assert results == [
            1,
            1,
            1,
            1,
        ], "did not split on all ones, got: {}".format(results)

    def test_random_generator(self):
        example = [1, 2, 3, 4]
        group_size = 2
        output = MTLDataset.random_generator(
            None, example, group_size=group_size
        )
        assert set(flatten(output)) == set(
            example
        ), "did not generate the exact set"
        assert (
            len(output) == len(example) // group_size
        ), "did not produce the right amount of groups"

        example = [1, 2, 3, 4]
        group_size = 2
        output = MTLDataset.random_generator(
            None, example, group_size=group_size, shuffle=False
        )
        assert flatten(output) == example, "did not generate the exact set"
        assert (
            len(output) == len(example) // group_size
        ), "did not produce the right amount of groups"

        example = [1, 2, 3, 4]
        group_size = 3
        output = MTLDataset.random_generator(
            None, example, group_size=group_size
        )
        assert (
            len(set(example).difference(set(flatten(output)))) == 1
        ), "did not generate the exact set"
        assert len(output) == math.floor(
            len(example) / group_size
        ), "did not produce the right amount of groups"
