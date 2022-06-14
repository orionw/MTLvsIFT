"""Tests for transferprediction.find_save_best_only.py"""

import logging
import os
import tempfile
import unittest
from shutil import rmtree
import random
import itertools
from argparse import Namespace
import json
import glob

import pandas as pd
import numpy as np
from transferprediction.find_save_best_only import save_and_remove_others
from transferprediction.settings import METRICS_OF_CHOICE, METRICS_FOR_TASK
from tests.settings import OUTPUT_FIXTURE_PATH_JSON


class CreateResultsCSVTestCase(unittest.TestCase):
    """Test transferprediction.find_save_best_only.create_results_csv_test_case."""

    def setUp(self):
        self.WORKING_DIR = os.path.join(OUTPUT_FIXTURE_PATH_JSON, "fake")
        self.TASK_NAMES = ["CoLA", "SST-2", "MRPC", "STS-B"]
        self.n_repeats = (
            15  # the number of repeats to do because of the randomness
        )

    def tearDown(self):
        self.delete_folder(self.WORKING_DIR)

    def delete_folder(self, folder_path: str):
        try:
            rmtree(folder_path)
        except Exception:
            pass
        assert not os.path.isdir(
            folder_path
        ), "did not delete the folder correctly, path={}".format(folder_path)

    def build_fake_checkpoints(
        self, dir_path: str, n_checkpoints: int, task_name_list: list
    ):
        """
        This is the main function to build the fake checkpoints and include the `results.json` file in the main folder
        so that we can test our script

        Args:
        -----
        dir_path: the directory to create fake sub-directories and store the `results.json` file
        n_checkpoints: an int indicating the amount of checkpoints to build
        task_name_list: a list of names (n: 1-inf) that tell what metrics to build in the `results.json` file

        Returns
        -------
        The checkpoint number and score of the best scoring value in `results.json`
        """
        max_score_index = random.sample(range(0, n_checkpoints + 1), 1)[0]
        max_score_index = 10
        max_score = random.uniform(0.1, 1)
        random_checkpoints = random.sample(range(0, 999999), n_checkpoints)
        assert (
            len(random_checkpoints) == n_checkpoints == 10
        ), "wrong number of checkpoints to generate"
        assert (
            len(glob.glob(os.path.join(dir_path, "checkpoint-*"))) == 0
        ), "did not start with zero checkpoints"
        for index, checkpoint in enumerate(random_checkpoints):
            checkpoint_folder_path = os.path.join(
                dir_path, "checkpoint-" + str(checkpoint)
            )
            os.makedirs(checkpoint_folder_path)
            with open(
                os.path.join(checkpoint_folder_path, "pytorch_model.bin"), "w"
            ) as fout:
                fout.write("empty")

        assert (
            len(glob.glob(os.path.join(dir_path, "checkpoint-*")))
            == n_checkpoints
        ), "made the wrong amount of checkpoints"

        fake_json_results = self.create_fake_json_results(
            max_score, max_score_index, random_checkpoints, task_name_list
        )
        with open(os.path.join(dir_path, "results.json"), "w") as fout:
            fout.write(json.dumps(fake_json_results))

        with open(os.path.join(dir_path, "pytorch_model.bin"), "w") as fout:
            fout.write("empty")

        return (
            random_checkpoints[max_score_index]
            if max_score_index < n_checkpoints
            else task_name_list[0].upper(),
            max_score,
        )

    def create_fake_json_results(
        self,
        max_score: list,
        max_index: int,
        random_checkpoints: list,
        task_name_list: list,
    ):
        """
        This is a lot of build code to replicate a random instance of the output data
        I decided to go with random in order to more thoroughly stress test this
        It builds a dict with the results from each checkpoint, putting the max value in the correct checkpoint location
        """
        results_dict = {}
        for task in task_name_list:
            for checkpoint_index, checkpoint in enumerate(random_checkpoints):
                name = task + "_" + str(checkpoint)
                value_dict = {}
                for metric in METRICS_FOR_TASK[task.lower()]:
                    if metric == METRICS_OF_CHOICE[task.lower()]:
                        value_dict[metric] = (
                            random.uniform(0, max_score)
                            if checkpoint_index != max_index
                            else max_score
                        )
                    else:
                        value_dict[metric] = random.uniform(
                            0, 1
                        )  # we don't care about this one
                results_dict[name] = value_dict
            # repeat the cycle once more and add the task name, like it does in transformers
            value_dict = {}
            for metric in METRICS_FOR_TASK[task.lower()]:
                if metric == METRICS_OF_CHOICE[task.lower()]:
                    # +2 to shift the iterator by one and another because we're counting indexes +1
                    value_dict[metric] = (
                        random.uniform(0, max_score)
                        if (checkpoint_index + 1) != max_index
                        else max_score
                    )
                else:
                    value_dict[metric] = random.uniform(
                        0, 1
                    )  # we don't care about this one
            results_dict[task + "_" + task.upper()] = value_dict

        return results_dict

    def find_and_remove_checkpoints_single(self):
        best_dict = {}
        for task in self.TASK_NAMES:
            best_check, best_score = self.build_fake_checkpoints(
                os.path.join(self.WORKING_DIR, task), 10, [task]
            )
            best_dict[task] = {"checkpoint": best_check, "score": best_score}

        args = Namespace(
            file_dir=self.WORKING_DIR,
            remove_all=False,
            single=True,
            real_task=None,
        )
        save_and_remove_others(args)

        # check that everything is deleted
        for task in self.TASK_NAMES:
            n_checkpoints_left = len(
                glob.glob(os.path.join(self.WORKING_DIR, task, "checkpoint-*"))
            )
            assert (
                n_checkpoints_left <= 1
            ), "kept too many checkpoints, expected 1, got {} in {}".format(
                n_checkpoints_left, task
            )
            if n_checkpoints_left < 1:
                assert os.path.isfile(
                    os.path.join(self.WORKING_DIR, task, "pytorch_model.bin")
                ), "can't find model!"
            else:
                assert os.path.isfile(
                    os.path.join(
                        glob.glob(
                            os.path.join(self.WORKING_DIR, task, "checkpoint-*")
                        )[0],
                        "pytorch_model.bin",
                    )
                ), "can't find model!"

        # check that the numbers are right
        for task in self.TASK_NAMES:
            if type(best_dict[task]["checkpoint"]) == int:
                assert os.path.isdir(
                    os.path.join(
                        self.WORKING_DIR,
                        task,
                        "checkpoint-" + str(best_dict[task]["checkpoint"]),
                    )
                ), "folder for best checkpoint did not exist"
            else:
                assert (
                    len(
                        glob.glob(
                            os.path.join(self.WORKING_DIR, task, "checkpoint-*")
                        )
                    )
                    == 0
                ), "had more than one dir with no best checkpoint"
            results = pd.read_csv(
                os.path.join(self.WORKING_DIR, task, "results.csv"),
                header=0,
                index_col=0,
            )
            cur_metric = METRICS_OF_CHOICE[task.lower()]
            best_row = results.iloc[results[cur_metric].idxmax(), :]
            assert np.isclose(
                best_row[cur_metric], best_dict[task]["score"]
            ), "scores did not line up in results.csv"
            assert str(best_row["checkpoint"]) == str(
                best_dict[task]["checkpoint"]
            ), "checkpoints did not line up in results.csv"

    def find_and_remove_multi(self):
        best_dict = {}
        permutations = list(itertools.permutations(self.TASK_NAMES, 2))
        for permutation in permutations:
            best_check, best_score = self.build_fake_checkpoints(
                os.path.join(self.WORKING_DIR, "--".join(permutation)),
                10,
                list(permutation),
            )
            best_dict[permutation] = {
                "checkpoint": best_check,
                "score": best_score,
            }

        args = Namespace(
            file_dir=self.WORKING_DIR,
            remove_all=False,
            single=False,
            real_task=None,
        )
        save_and_remove_others(args)

        # check that everything is deleted
        for task_set in permutations:
            task_file_name = "--".join(task_set)
            n_checkpoints_left = len(
                glob.glob(
                    os.path.join(
                        self.WORKING_DIR, task_file_name, "checkpoint-*"
                    )
                )
            )
            assert (
                n_checkpoints_left <= 1 or not best_dict[task_set].isnumeric()
            ), "kept too many checkpoints, expected 1, got {} in {}".format(
                n_checkpoints_left, task_set
            )
            if n_checkpoints_left < 1:
                assert os.path.isfile(
                    os.path.join(
                        self.WORKING_DIR, task_file_name, "pytorch_model.bin"
                    )
                ), "can't find model!"
            else:
                assert os.path.isfile(
                    os.path.join(
                        glob.glob(
                            os.path.join(
                                self.WORKING_DIR, task_file_name, "checkpoint-*"
                            )
                        )[0],
                        "pytorch_model.bin",
                    )
                ), "can't find model!"

        # check that the numbers are right
        for task_set in permutations:
            task_file_name = "--".join(task_set)
            if type(best_dict[task_set]["checkpoint"]) == int:
                assert os.path.isdir(
                    os.path.join(
                        self.WORKING_DIR,
                        task_file_name,
                        "checkpoint-" + str(best_dict[task_set]["checkpoint"]),
                    )
                ), "folder for best checkpoint did not exist"
            else:
                assert (
                    len(
                        glob.glob(
                            os.path.join(
                                self.WORKING_DIR, task_file_name, "checkpoint-*"
                            )
                        )
                    )
                    == 0
                ), "had more than one dir when there should be no best checkpoint at {}".format(
                    os.path.join(
                        self.WORKING_DIR, task_file_name, "checkpoint-*"
                    )
                )
            results = pd.read_csv(
                os.path.join(self.WORKING_DIR, task_file_name, "results.csv"),
                header=0,
                index_col=0,
            )
            cur_metric = METRICS_OF_CHOICE[task_set[0].lower()]
            best_row = results.loc[
                results[results["task"] == task_set[0]][cur_metric].idxmax()
            ]
            assert np.isclose(
                best_row[cur_metric], best_dict[task_set]["score"]
            ), "scores did not line up in results.csv"
            assert str(best_row["checkpoint"]) == str(
                best_dict[task_set]["checkpoint"]
            ), "checkpoints did not line up in results.csv"

    def find_and_remove_all_multi(self):
        best_dict = {}
        permutations = list(itertools.permutations(self.TASK_NAMES, 2))
        for permutation in permutations:
            best_check, best_score = self.build_fake_checkpoints(
                os.path.join(self.WORKING_DIR, "--".join(permutation)),
                10,
                list(permutation),
            )
            best_dict[permutation] = {
                "checkpoint": best_check,
                "score": best_score,
            }

        args = Namespace(
            file_dir=self.WORKING_DIR,
            remove_all=True,
            single=False,
            real_task=None,
        )
        save_and_remove_others(args)

        # check that everything is deleted
        for task_set in permutations:
            task_file_name = "--".join(task_set)
            n_checkpoints_left = len(
                glob.glob(
                    os.path.join(
                        self.WORKING_DIR, task_file_name, "checkpoint-*"
                    )
                )
            )
            assert (
                n_checkpoints_left == 0
            ), "kept too many checkpoints, expected 0, got {} in {}".format(
                n_checkpoints_left, task_set
            )
            assert not os.path.isfile(
                os.path.join(
                    self.WORKING_DIR, task_file_name, "pytorch_model.bin"
                )
            ), "can find model, error!"

        # check that the numbers are right
        for task_set in permutations:
            task_file_name = "--".join(task_set)
            assert (
                len(
                    glob.glob(
                        os.path.join(
                            self.WORKING_DIR, task_file_name, "checkpoint-*"
                        )
                    )
                )
                == 0
            ), "had more than one dir when removing all"
            results = pd.read_csv(
                os.path.join(self.WORKING_DIR, task_file_name, "results.csv"),
                header=0,
                index_col=0,
            )
            cur_metric = METRICS_OF_CHOICE[task_set[0].lower()]
            best_row = results.loc[
                results[results["task"] == task_set[0]][cur_metric].idxmax()
            ]
            assert np.isclose(
                best_row[cur_metric], best_dict[task_set]["score"]
            ), "scores did not line up in results.csv"
            assert str(best_row["checkpoint"]) == str(
                best_dict[task_set]["checkpoint"]
            ), "checkpoints did not line up in results.csv"

    def find_and_remove_all_single(self):
        best_dict = {}
        for task in self.TASK_NAMES:
            best_check, best_score = self.build_fake_checkpoints(
                os.path.join(self.WORKING_DIR, task), 10, [task]
            )
            best_dict[task] = {"checkpoint": best_check, "score": best_score}

        args = Namespace(
            file_dir=self.WORKING_DIR,
            remove_all=True,
            single=True,
            real_task=None,
        )
        save_and_remove_others(args)

        # check that everything is deleted
        for task in self.TASK_NAMES:
            n_checkpoints_left = len(
                glob.glob(os.path.join(self.WORKING_DIR, task, "checkpoint-*"))
            )
            assert (
                n_checkpoints_left == 0
            ), "kept too many checkpoints, expected 0, got {} in {}".format(
                n_checkpoints_left, task
            )
            assert not os.path.isfile(
                os.path.join(self.WORKING_DIR, task, "pytorch_model.bin")
            ), "can find model, error!"

        # check that the numbers are right
        for task in self.TASK_NAMES:
            assert (
                len(
                    glob.glob(
                        os.path.join(self.WORKING_DIR, task, "checkpoint-*")
                    )
                )
                == 0
            ), "had more than one dir when removing all"
            results = pd.read_csv(
                os.path.join(self.WORKING_DIR, task, "results.csv"),
                header=0,
                index_col=0,
            )
            cur_metric = METRICS_OF_CHOICE[task.lower()]
            best_row = results.iloc[results[cur_metric].idxmax(), :]
            assert np.isclose(
                best_row[cur_metric], best_dict[task]["score"]
            ), "scores did not line up in results.csv"
            assert str(best_row["checkpoint"]) == str(
                best_dict[task]["checkpoint"]
            ), "checkpoints did not line up in results.csv"

    def test_single_n_times(self):
        # due to randomness, run this a bunch
        for i in range(self.n_repeats):
            self.find_and_remove_checkpoints_single()

    def test_multi_n_times(self):
        # due to randomness, run this a bunch
        for i in range(self.n_repeats):
            self.find_and_remove_multi()
            self.tearDown()
            assert not os.path.isdir(self.WORKING_DIR)

    def test_multi_remove_all_n_times(self):
        # due to randomness, run this a bunch
        for i in range(self.n_repeats):
            self.find_and_remove_all_multi()
            self.tearDown()

    def test_single_remove_all_n_times(self):
        # due to randomness, run this a bunch
        for i in range(self.n_repeats):
            self.find_and_remove_all_single()
