import os
import glob
import csv
import numpy as np
import pandas as pd
import random
from shutil import copyfile

from transferprediction.io_utils import load_glue

random.seed(42)
np.random.seed(42)


pairs_to_run = [
    ("QNLI", "MNLI"),  # example in paper
]


def save_test_and_dev_unchanged(src, dst_train, is_mnli: bool):
    splits = ["test", "dev"]
    if is_mnli:
        splits_new = [split_val + "_mismatched" for split_val in splits]
        splits_new.extend([split_val + "_matched" for split_val in splits])
        splits = splits_new
    for split_type in splits:
        copyfile(
            src.replace("train", split_type),
            dst_train.replace("train", split_type),
        )


def create_size_datasets(data_folder="glue_data"):
    """
    Goes through all pairs of `original_sizes` dataset and creates a primary, with the secondary having a dataset
        of various sizes that are close and around to the size of the primary for testing
    """
    amounts_around = [0.33, 0.5, 1, 2, 3]
    root_dir = "glue_size_datasets"
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    for divide_from_starting_place in [1, 0.5, 0.33]:
        print(f"On divide by {divide_from_starting_place}")
        for (target_dataset_name, supporting_dataset_name) in pairs_to_run:
            print(f"On primary dataset {target_dataset_name}")
            df_primary = load_glue(target_dataset_name, data_folder, "train")
            primary_dataset_size = int(
                len(df_primary) * divide_from_starting_place
            )
            df_primary = df_primary.sample(n=primary_dataset_size)
            print(
                f"Sampling to {primary_dataset_size} for {target_dataset_name}"
            )

            if not os.path.isdir(
                os.path.join(
                    root_dir,
                    target_dataset_name + "_" + str(divide_from_starting_place),
                )
            ):
                os.makedirs(
                    os.path.join(
                        root_dir,
                        target_dataset_name
                        + "_"
                        + str(divide_from_starting_place),
                    )
                )

            save_test_and_dev_unchanged(
                os.path.join(data_folder, target_dataset_name, "train.tsv"),
                os.path.join(
                    root_dir,
                    target_dataset_name + "_" + str(divide_from_starting_place),
                    "train" + ".tsv",
                ),
                is_mnli=target_dataset_name == "MNLI",
            )

            df_primary.to_csv(
                os.path.join(
                    root_dir,
                    target_dataset_name + "_" + str(divide_from_starting_place),
                    "train" + ".tsv",
                ),
                quoting=csv.QUOTE_NONE,
                sep="\t",
                escapechar="\\",
                index=None,
                float_format="%.0f",
            )

            df_support = load_glue(
                supporting_dataset_name, data_folder, "train"
            )
            for amount_to_divide_by in amounts_around:
                size_to_sample_to = int(len(df_primary) // amount_to_divide_by)
                print(
                    f"Sampling to {size_to_sample_to} for {supporting_dataset_name}"
                )
                df_support_current = df_support.sample(n=size_to_sample_to)

                path_to_cur_supporting = os.path.join(
                    root_dir,
                    supporting_dataset_name
                    + f"_{divide_from_starting_place}_{amount_to_divide_by}",
                )

                if not os.path.isdir(path_to_cur_supporting):
                    os.makedirs(path_to_cur_supporting)

                save_test_and_dev_unchanged(
                    os.path.join(
                        data_folder, supporting_dataset_name, "train.tsv"
                    ),
                    os.path.join(
                        path_to_cur_supporting,
                        "train" + ".tsv",
                    ),
                    is_mnli=supporting_dataset_name == "MNLI",
                )

                df_support_current.to_csv(
                    os.path.join(
                        path_to_cur_supporting,
                        "train" + ".tsv",
                    ),
                    quoting=csv.QUOTE_NONE,
                    sep="\t",
                    escapechar="\\",
                    index=None,
                    float_format="%.0f",
                )


if __name__ == "__main__":
    create_size_datasets()
