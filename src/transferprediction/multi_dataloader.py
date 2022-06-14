import torch
import decimal
import argparse
import random
import itertools
import typing
import math
import warnings
from torch._six import int_classes as _int_classes
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    DistributedSampler,
    Dataset,
    Sampler,
)
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from transferprediction.settings import METRICS_OF_CHOICE
from transferprediction.utils import flatten


def empty_zip_longest(data_to_zip):
    """Zips longest but is empty in place of `None"""
    new_data = itertools.zip_longest(*data_to_zip)
    final_data = []
    for group in new_data:
        delete_indexes = []
        for index, group_item in enumerate(group):
            if group_item is None:
                delete_indexes.append(index)
        final_data.append(
            [
                group[group_index]
                for group_index in range(len(group))
                if group_index not in delete_indexes
            ]
        )
    return final_data


class MTLDataset(Dataset):
    """
    Uses the `random` module for randomness.  Set the seed before calling this class.
    NOTE: always `drops_last` in dataloader terms.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        dataset_list: list,
        batch_size: int,
        single_task_scores: typing.List[float] = None,
    ):
        self.args = args
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.single_task_scores = single_task_scores

        if args.sampling_type == "dynamic" and single_task_scores is None:
            raise Exception(
                "Need single task scores to compute dynamic sampling"
            )

        for dataset in dataset_list:
            if type(dataset) != torch.utils.data.dataset.TensorDataset:
                raise NotImplementedError(
                    "Don't have batching implemented for dataset type {},  Use TensorDataset instead".format(
                        type(dataset)
                    )
                )

        self.all_instances = self.get_all_instances()

        # these should be initialized every epoch by the MTLRandomSampler
        self.instances = None
        self.arranged_instances = None
        self.instances_per_batch = None
        self.min_length = None
        self.evaluation_metrics = None
        self.old_evaluation_metrics = False

        # do a first initialization to get length initialized
        if (
            args.sampling_type == "dynamic"
        ):  # dynamic starts by size initialization, can change later
            self.instances = self.arrange_instances_by_size()
            self.use_size_first = True
        else:
            self.use_size_first = False
            self.sample_batches()

        self.arrange_batches()

    def __len__(self):
        return len(self.arranged_instances)

    def __getitem__(
        self,
        idx,
        evaluation_metrics: typing.Dict[str, typing.Dict[str, float]] = None,
    ):
        return self.arranged_instances[idx]

    def get_all_instances(self):
        data = []
        for index, dataset in enumerate(self.dataset_list):
            cur_tensors = dataset.tensors
            if len(cur_tensors) == 0:
                raise ValueError("Dataset had zero length tensors")
            chunked_data = [
                torch.chunk(tensor_item, tensor_item.shape[0])
                for tensor_item in cur_tensors
            ]
            all_tensors = data.append(
                [tensor + (index,) for tensor in zip(*chunked_data)]
            )
        return data

    def sample_batches(self):
        if self.args.sampling_type == "size":
            self.instances = self.arrange_instances_by_size()
        elif self.args.sampling_type == "uniform":
            self.instances = self.arrange_instances_uniform()
        elif self.args.sampling_type == "dynamic":
            if self.evaluation_metrics is None or self.old_evaluation_metrics:
                raise Exception(
                    "Using dynamic sampling without giving evaluation metrics"
                )
            self.instances = self.arrange_instances_dynamically(
                self.evaluation_metrics
            )
            self.old_evaluation_metrics = True
        elif self.args.sampling_type == "all":
            self.instances = self.all_instances
        else:
            raise NotImplementedError(
                "Don't know how to process `sampling_type`={}".format(
                    self.args.sampling_type
                )
            )
        print("Done!")

    def arrange_batches(self):
        if self.args.batch_type == "partition":
            self.arranged_instances = self.group_instances_by_partioning()
        elif self.args.batch_type == "homogeneous":
            self.arranged_instances = self.group_instances_homogeneous()
        elif self.args.batch_type == "heterogeneous":
            self.arranged_instances = self.group_instances_heterogeneous()
        elif self.args.batch_type == "forced_heterogeneous":
            self.arranged_instances = (
                self.group_instances_forced_heterogeneous()
            )
        elif self.args.sampling_type == "all":
            self.arranged_instances = flatten(self.all_instances)
        else:
            raise NotImplementedError(
                "Don't know how to process `batch_type`={}".format(
                    self.args.batch_type
                )
            )

    def convert_heterogeneous_batches_to_homogenous(
        self, instances: typing.List[typing.List[typing.List]]
    ) -> typing.List[typing.List]:
        """
        A helper function to convert the batches created by the `sampling_type` (where each batch contains all tasks)
        to `N` lists containing instances for the `N`-th task

        Used by the `batch_type` functions
        """
        per_dataset_instances = []
        for index in range(len(self.dataset_list)):
            dataset_items = [item[index] for item in instances]
            flat_items = flatten(dataset_items)
            per_dataset_instances.append(flat_items)

        return per_dataset_instances

    def group_instances_homogeneous(self):
        """
        Homogeneous Batching means that each batch will contain only one task, but the order of the batches are shuffled.
        Thus, each batch will be homogeneous since it will have only one task but the ordering of those batches
        will be random.

        Returns:
        --------
            A list of instances where each `self.batch_size` of the list contains instances from only one task
            but the ordering of those batches is random
        """
        all_batches_homogeneous = []
        per_task_instances = self.convert_heterogeneous_batches_to_homogenous(
            self.instances
        )
        for task_instances in per_task_instances:
            groups = self.random_generator(
                self,
                task_instances,
                group_size=self.batch_size,
                shuffle=False,  # already shuffled in `sampling_type` functions
            )
            all_batches_homogeneous.extend(groups)
        random.shuffle(all_batches_homogeneous)  # shuffle the batch orders
        final_instances = flatten(
            all_batches_homogeneous
        )  # unflatten the batches
        if len(final_instances) == 0:
            raise ValueError(
                "zero length instance array, perhaps the batch size is greater than the number of instances?"
            )
        return final_instances

    def group_instances_heterogeneous(self):
        """
        Heterogeneous Batching (random) means that we will take all instances from all tasks and shuffle them.
        We will then have heterogeneous batches because each batch will have a random amount of samples

        Returns:
        --------
            A list of instances where each `self.batch_size` of the list contains random instances
        """
        per_task_instances = self.convert_heterogeneous_batches_to_homogenous(
            self.instances
        )
        all_instances = flatten(per_task_instances)
        random.shuffle(all_instances)  # we want random instances
        if len(all_instances) == 0:
            raise ValueError(
                "zero length instance array, perhaps the batch size is greater than the number of instances?"
            )
        return all_instances

    def group_instances_forced_heterogeneous(self):
        """
        Forced Heterogeneous Batching means that each batch will be forced to have each task dataset in each batch.
        Thus, each batch will be heterogeneous because we force some samples into each batch.  Since
        `self.instances` is already structured in batches with each dataset as lists within each batch
        simply shuffle the order they appear in the batch and add it

        Returns:
        --------
            A list of instances where each `self.batch_size` of the list contains instances from each task
        """
        final_instances = []
        for batch in self.instances:
            random.shuffle(batch)  # shuffle the order each time
            flat_batch = flatten(batch)  # merge in the N dataset lists
            final_instances.extend(flat_batch)
        if len(final_instances) == 0:
            raise ValueError(
                "zero length instance array, perhaps the batch size is greater than the number of instances?"
            )
        return final_instances

    def group_instances_by_partioning(self):
        """
        Paritioned Batching means that each task will appear in order, in their corresponding batches
        Thus, we will add all of task 1 then task 2 and so on.

        Returns:
        --------
            A list of instances where each task instances precede the following task instances
        """
        per_task_instances = self.convert_heterogeneous_batches_to_homogenous(
            self.instances
        )
        all_instances_partitioned = flatten(per_task_instances)
        if len(all_instances_partitioned) == 0:
            raise ValueError(
                "zero length instance array, perhaps the batch size is greater than the number of instances?"
            )
        return all_instances_partitioned

    @staticmethod
    def random_generator(
        self, seq: list, group_size: int, shuffle: bool = True
    ):
        rand_seq = seq[:]  # make a copy to avoid changing input argument
        if shuffle:
            random.shuffle(rand_seq)
        lists = []
        limit = len(rand_seq) - 1
        for i, group in enumerate(zip((*[iter(rand_seq)] * group_size))):
            delete_indexes = []
            for index, group_item in enumerate(group):
                if group_item is None:
                    delete_indexes.append(index)

            group = [
                group[group_index]
                for group_index in range(len(group))
                if group_index not in delete_indexes
            ]
            lists.append(group)
            if i == limit:
                break  # have enough
        return lists

    def get_instances_numbers_from_proportions(self, proportions: typing.List):
        """A helper function to get the number of instances per batch given proportions"""
        if not np.isclose(sum(proportions), 1.0):
            raise Exception(
                "Got proportions that do not add up to 1.0: {}, sum={}".format(
                    proportions, sum(proportions)
                )
            )
        if self.batch_size < len(proportions):
            raise Exception(
                "Cannot evenly split into proportions when the number of dataset is greater than the batch_size"
            )
        if self.batch_size == len(proportions):
            return [1] * len(proportions)

        instances_per_batch = [
            int(round(prop * self.batch_size)) for prop in proportions
        ]
        instances_per_batch_decimal = [
            int(math.modf(prop * self.batch_size)[0]) for prop in proportions
        ]

        # make sure the amount is equal to the correct batch size for size and dynamic sampling
        if self.args.sampling_type != "uniform":
            while self.batch_size - sum(instances_per_batch) > 0:
                max_index = np.argmax(np.array(instances_per_batch_decimal))
                instances_per_batch[max_index] += 1

            while sum(instances_per_batch) - self.batch_size > 0:
                min_index = np.argmin(np.array(instances_per_batch_decimal))
                instances_per_batch[min_index] -= 1

        # make sure none are zero
        while 0 in instances_per_batch:
            cur_zero_index = instances_per_batch.index(0)
            max_index = instances_per_batch.index(max(instances_per_batch))
            instances_per_batch[max_index] -= 1
            instances_per_batch[cur_zero_index] += 1

        assert (
            0 not in instances_per_batch
        ), "got zero length batch size for task: {}".format(instances_per_batch)

        if self.args.sampling_type != "uniform":
            assert (
                abs(sum(instances_per_batch) - self.batch_size) <= 1.0
            ), "rounding did not work: {} vs batch {}".format(
                sum(instances_per_batch), self.batch_size
            )
        else:
            # uniform can be more inefficient to make sure it's uniform
            assert (
                abs(sum(instances_per_batch) - self.batch_size)
                <= len(instances_per_batch) - 1
            ), "rounding did not work: {} vs batch {}".format(
                sum(instances_per_batch), self.batch_size
            )
        return instances_per_batch

    def gather_last_batch(
        self,
        sampled_bunches: typing.List[typing.List],
    ) -> typing.List:
        last_batch = [
            dataset_batches[-1] for dataset_batches in sampled_bunches
        ]
        if self.args.sampling_type == "uniform":
            pass
        elif self.args.sampling_type in ["dynamic", "size"]:
            pass
        else:
            raise NotImplementedError(
                "Can't parse sampling type {}".format(self.args.sampling_type)
            )

    def sample_from_all_instances(
        self, number_to_sample_per_dataset: typing.List[int]
    ):
        """A helper function to get batches from all instances randomly using the numbers in `number_to_sample_per_dataset`"""
        sampled_bunches: typing.List[
            typing.List[typing.Tuple]
        ] = (
            []
        )  # each tuple will contains 4 tensors and one int indicating which dataset
        for index, dataset_instances in enumerate(self.all_instances):
            if number_to_sample_per_dataset[index] == 0:
                continue
            groups = self.random_generator(
                self,
                dataset_instances,
                group_size=number_to_sample_per_dataset[index],
            )
            sampled_bunches.append(groups)

        # this shortens them all to the same length groups and combines them in batches
        # last_batch = self.gather_last_batch(sampled_bunches, number_to_sample_per_dataset)
        grouped_samples = list(zip(*sampled_bunches))
        assert len(grouped_samples) != 0, "have zero length samples in groups"
        # assert len(grouped_samples) == max([len(batch) for batch in sampled_bunches]), "did not create an equal amount of batches"
        return grouped_samples

    def arrange_instances_by_size(self):
        """
        Gets the batches for each dataset arranged by the size of the datasets.  Uses the proportional size of each dataset.

        Gets the number of instances for each dataset which should be in each batch.  It then uses that to create batches
        where each dataset is still in a seperate sub-list

        Returns:
        --------
            a list containing a list for each batch where each batch-list contains a list of instances for the respective dataset index
        """
        if self.instances_per_batch is None:
            size_proportions = np.array(
                [len(dataset.tensors[0]) for dataset in self.dataset_list]
            )
            size_proportions = size_proportions / np.array(
                sum(size_proportions)
            )
            self.instances_per_batch = (
                self.get_instances_numbers_from_proportions(size_proportions)
            )
        assert (
            sum(self.instances_per_batch) == self.batch_size
        ), "rounding did not work"
        grouped_samples = self.sample_from_all_instances(
            self.instances_per_batch
        )
        return grouped_samples

    def arrange_instances_uniform(self):
        """
        Gets the batches for each dataset arranged uniformly, each getting (batch_size / # of datasets) samples, or the size of the smallest dataset

        Returns:
        --------
            a list containing a list for each batch where each batch-list contains a list of instances for the respective dataset index
        """
        if self.instances_per_batch is None:
            proportions = 1.0 / len(self.dataset_list)
            self.instances_per_batch = (
                self.get_instances_numbers_from_proportions(
                    [proportions] * len(self.dataset_list)
                )
            )

        assert (
            len(set(self.instances_per_batch)) == 1
        ), "is not uniform, {}".format(self.instances_per_batch)
        assert (
            (self.batch_size - len(self.dataset_list))
            <= sum(self.instances_per_batch)
            <= self.batch_size
        ), "rounding did not work"

        grouped_samples = self.sample_from_all_instances(
            self.instances_per_batch
        )
        return grouped_samples

    def arrange_instances_dynamically(
        self, evaluation_metrics: typing.Dict[str, typing.Dict[str, float]]
    ):
        """
        Uses the scores from the previous epoch to determine the sampling strategy for this epoch.
        From the paper "Dynamic Sampling Strategies for Multi-Task Reading Comprehension"

        Finds the difference between the single task scores and the multi-task scores and uses that to
        create the sampling distribution.

        Returns:
        --------
            a list containing a list for each batch where each batch-list contains a list of instances for the respective dataset index
        """
        sampling_probability = self.get_dynamic_sampling_distribution(
            evaluation_metrics
        )
        print(
            "\n\n ### Dynamic Sampling is now at {} ###\n\n".format(
                sampling_probability
            )
        )
        self.instances_per_batch = self.get_instances_numbers_from_proportions(
            sampling_probability
        )
        grouped_samples = self.sample_from_all_instances(
            self.instances_per_batch
        )
        return grouped_samples

    def get_dynamic_sampling_distribution(
        self, evaluation_metrics: typing.Dict[str, typing.Dict[str, float]]
    ) -> typing.List:
        """Given the evaluation metrics, calulate the sampling probability for each task compared to single-task scores"""
        task_scores = []
        for index, task in enumerate(self.args.task_names):
            metric_to_use = METRICS_OF_CHOICE[task.lower()]
            score = evaluation_metrics[task][metric_to_use]
            task_scores.append(score)
        task_scores = np.array(task_scores)
        # get how far away they are from the single task scores and normalize to get sampling dist
        diff_from_single = self.single_task_scores - task_scores
        sampling_probability = F.softmax(
            torch.tensor(diff_from_single),  # pylint: disable=not-callable
            dim=0,
        ).numpy()
        return sampling_probability

    def upsample_secondary(self) -> list:
        """
        Upsamples secondary datasets to be the same size as the first
        """
        dataset_list = self.dataset_list
        if len(dataset_list) == 1:
            return dataset_list

        new_list = [dataset_list[0]]
        first_len = len(dataset_list[0])
        for index, dataset in enumerate(dataset_list):
            if not index:
                continue
            instances_to_add = first_len - len(dataset)
            tensor_list = [dataset.tensors]
            while instances_to_add > 0:
                if instances_to_add > len(dataset):
                    tensor_list.append(dataset.tensors)
                    instances_to_add -= len(dataset.tensors[0])
                else:
                    tensor_list.append(
                        tuple(
                            [
                                tensor[:instances_to_add]
                                for tensor in dataset.tensors
                            ]
                        )
                    )
                    instances_to_add -= instances_to_add
            new_list.append(
                TensorDataset(
                    *[
                        torch.cat(tensors, dim=0)
                        for tensors in zip(*tensor_list)
                    ]
                )
            )
        return new_list


class MTLRandomSampler(Sampler):
    """
    A MTL Random Sampler
    Template taken from:
    https://github.com/pytorch/pytorch/blob/e870a9a87042805cd52973e36534357f428a0748/torch/utils/data/sampler.py#L68-L110
    and
    https://github.com/pytorch/pytorch/blob/e870a9a87042805cd52973e36534357f428a0748/torch/utils/data/sampler.py#L51-L65

    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        # resets and randomizes the instances according to the sampling and batch types
        if (
            self.data_source.args.sampling_type == "dynamic"
            and self.data_source.use_size_first
        ):  # dynamic starts by size initialization, can change later
            self.data_source.instances = (
                self.data_source.arrange_instances_by_size()
            )
            self.data_source.use_size_first = False
            warnings.warn(
                "### Using size for this iteration - should only happen at the beginning ###"
            )
        else:
            self.data_source.sample_batches()

        self.data_source.arrange_batches()
        # return SequentialSampler __iter__ since we already randomized
        return iter(range(len(self.data_source)))
