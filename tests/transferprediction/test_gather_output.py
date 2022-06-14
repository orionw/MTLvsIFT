"""Tests for transferprediction.create_data_matrix."""

import logging
import os
import tempfile
import unittest
from shutil import rmtree
from argparse import Namespace

import pandas as pd
import numpy as np
from transferprediction.create_data_matrix import (
    create_single_matrix,
    create_multi_matrix,
)
from tests.settings import OUTPUT_FIXTURE_PATH_CSV
from tests.utils import row_in_df


class CreateDataMatrixTestCase(unittest.TestCase):
    """Test transferprediction.create_data_matrix.create_single_matrix."""

    def setUp(self):
        self.OUTPUT_DIR = os.path.join("tests", "results")

    def tearDown(self):
        self.delete_folder(self.OUTPUT_DIR)

    def create_and_check_single_matrix(self):
        namespace_dict = {
            "dir_path": os.path.join(OUTPUT_FIXTURE_PATH_CSV, "single"),
            "output_dir": os.path.join(self.OUTPUT_DIR, "single"),
            "single": True,
            "use_seed": True,
            "single_full_data_path": None,
        }
        self.args = Namespace(**namespace_dict)
        create_single_matrix(self.args)
        assert os.path.isfile(
            os.path.join(self.OUTPUT_DIR, "single", "mean_single_pairs.csv")
        ), "did not create the correct file"

    def delete_folder(self, folder_path: str):
        try:
            rmtree(folder_path)
        except Exception:
            pass
        assert not os.path.isdir(
            folder_path
        ), "did not delete the folder correctly, path={}".format(folder_path)

    def test_create_data_matrix_creates_file(self):
        self.create_and_check_single_matrix()

    def test_create_data_matrix_gets_correct_values(self):
        self.create_and_check_single_matrix()

        # check mean aggregation correct
        mean = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "single", "mean_single_pairs.csv"),
            header=0,
            index_col=0,
        )
        correct_array = np.array([0.5, 0.55])
        assert (
            correct_array == mean["score"]
        ).all(), "did not get the correct output for mean"

        # check that it is reading the files correctly
        all_single_runs = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "single", "all_single_pairs.csv"),
            header=0,
        )
        assert row_in_df(
            pd.Series(["mnli", 1, 1.0, "seed-0"]), all_single_runs
        )  # mnli check
        assert row_in_df(
            pd.Series(["sts-b", 2, 0.1, "seed-1"]), all_single_runs
        )  # sts-b check
        assert row_in_df(
            pd.Series(["sts-b", 0, 1.0, "seed-0"]), all_single_runs
        )  # tie check

    def test_create_data_matrix_multi_correct_values_no_seeds(self):
        self.create_and_check_single_matrix()  # needed for multi matrix
        namespace_dict = {
            "dir_path": os.path.join(OUTPUT_FIXTURE_PATH_CSV, "multi"),
            "output_dir": os.path.join(self.OUTPUT_DIR, "multi"),
            "single_data_path": os.path.join(self.OUTPUT_DIR, "single/"),
            "single": False,
            "use_seed": False,
            "aggregate": "mean",
            "single_data_type": "mean",
            "dont_validate": True,
            "single_full_data_path": None,
        }
        self.args = Namespace(**namespace_dict)
        create_multi_matrix(self.args)
        assert os.path.isfile(
            os.path.join(self.OUTPUT_DIR, "multi", "matrix_prop_diff.csv")
        ), "did not create the matrix file"

        # check pair aggregation correct
        all_pairs = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "multi", "all_pairs.csv"), header=0
        )
        assert row_in_df(
            pd.Series(["mnli", "STS-B", 1, 1.0]), all_pairs
        )  # mnli primary check
        assert row_in_df(
            pd.Series(["sts-b", "MNLI", 0, 1.0]), all_pairs
        )  # sst-b primary check

        # check the matrix generation
        matrix = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "multi", "matrix_prop_diff.csv"),
            header=0,
        )
        matrix = matrix.fillna(-1)  # since dealing with NaNs is hard with ==
        # match columns and row task names
        columns = list(matrix.columns)
        columns.remove("primary")
        assert (
            list(matrix["primary"].values) == columns
        ), "did not contain the correct columns"
        # check data in matrix
        assert row_in_df(
            pd.Series(["MNLI", -1.0, 1.0]), matrix
        )  # MNLI matrix diff, 100% increase
        assert row_in_df(
            pd.Series(["STS-B", round(0.45 / 0.55, 15), -1.0]), matrix
        )  # STS-B matrix check, 82% increase

    def test_create_data_matrix_multi_correct_values_no_seeds_double_output(
        self,
    ):
        self.create_and_check_single_matrix()  # needed for multi matrix
        namespace_dict = {
            "dir_path": os.path.join(OUTPUT_FIXTURE_PATH_CSV, "multi_doubles"),
            "output_dir": os.path.join(self.OUTPUT_DIR, "multi_doubles"),
            "single_data_path": os.path.join(self.OUTPUT_DIR, "single/"),
            "single": False,
            "use_seed": False,
            "aggregate": "mean",
            "single_data_type": "mean",
            "dont_validate": True,
            "single_full_data_path": None,
        }
        self.args = Namespace(**namespace_dict)
        create_multi_matrix(self.args)
        assert os.path.isfile(
            os.path.join(
                self.OUTPUT_DIR, "multi_doubles", "matrix_prop_diff.csv"
            )
        ), "did not create the matrix file"

        # check pair aggregation correct
        all_pairs = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "multi_doubles", "all_pairs.csv"),
            header=0,
        )
        assert row_in_df(
            pd.Series(["mnli", "STS-B", 1, 1.0]), all_pairs
        )  # mnli primary check
        assert row_in_df(
            pd.Series(["sts-b", "MNLI", 0, 1.0]), all_pairs
        )  # sst-b primary check

        # check the matrix generation
        matrix = pd.read_csv(
            os.path.join(
                self.OUTPUT_DIR, "multi_doubles", "matrix_prop_diff.csv"
            ),
            header=0,
        )
        matrix = matrix.fillna(-1)  # since dealing with NaNs is hard with ==
        # match columns and row task names
        columns = list(matrix.columns)
        columns.remove("primary")
        assert (
            list(matrix["primary"].values) == columns
        ), "did not contain the correct columns"
        # check data in matrix
        assert row_in_df(
            pd.Series(["MNLI", -1.0, 1.0]), matrix
        )  # MNLI matrix diff, 100% increase
        assert row_in_df(
            pd.Series(["STS-B", round(0.45 / 0.55, 15), -1.0]), matrix
        )  # STS-B matrix check, 82% increase

    def test_create_data_matrix_multi_correct_values_with_seeds(self):
        self.create_and_check_single_matrix()  # needed for multi matrix
        namespace_dict = {
            "dir_path": os.path.join(OUTPUT_FIXTURE_PATH_CSV, "multi_seed"),
            "output_dir": os.path.join(self.OUTPUT_DIR, "multi_seed"),
            "single_data_path": os.path.join(self.OUTPUT_DIR, "single/"),
            "single": False,
            "use_seed": True,
            "aggregate": "mean",
            "single_data_type": "mean",
            "dont_validate": True,
            "single_full_data_path": None,
        }
        self.args = Namespace(**namespace_dict)
        create_multi_matrix(self.args)
        assert os.path.isfile(
            os.path.join(self.OUTPUT_DIR, "multi_seed", "matrix_prop_diff.csv")
        ), "did not create the matrix file"

        # check pair aggregation correct
        all_pairs = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "multi_seed", "all_pairs.csv"),
            header=0,
        )
        assert row_in_df(
            pd.Series(["mnli", "STS-B", 1.0, 0.0]), all_pairs
        )  # mnli primary check
        assert row_in_df(
            pd.Series(["sts-b", "MNLI", 1.0, 0.0]), all_pairs
        )  # sst-b primary check

        # check the matrix generation
        matrix = pd.read_csv(
            os.path.join(self.OUTPUT_DIR, "multi_seed", "matrix_prop_diff.csv"),
            header=0,
        )
        matrix = matrix.fillna(-1)  # since dealing with NaNs is hard with ==
        # match columns and row task names
        columns = list(matrix.columns)
        columns.remove("primary")
        assert (
            list(matrix["primary"].values) == columns
        ), "did not contain the correct columns"
        # check data in matrix
        assert row_in_df(
            pd.Series(["MNLI", -1.0, 1.0]), matrix
        )  # MNLI matrix diff, 100% increase
        assert row_in_df(
            pd.Series(["STS-B", round(0.45 / 0.55, 15), -1.0]), matrix
        )  # STS-B matrix check, 82% increase
