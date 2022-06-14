import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from transferprediction.settings import METRICS_OF_CHOICE, LOWER_TO_UPPER_TASKS


def gather_validation_exp(args: argparse.Namespace):
    results_list = []
    dont_haves = {}
    for seed_path in glob.glob(os.path.join(args.data_dir, "*")):
        seed_num = seed_path.split("/")[-1]
        for prop_path in glob.glob(os.path.join(seed_path, "*")):
            prop_main = prop_path.split("/")[-1]
            for task_run_path in glob.glob(os.path.join(prop_path, "*")):
                task_full = task_run_path.split("/")[-1]
                first_task, second_task = task_full.split("--")
                prop_support = second_task.split("_")[-1]
                prop_target = first_task.split("_")[-1]
                target_task = first_task.split("_")[0]
                try:
                    all_results = pd.read_csv(
                        os.path.join(task_run_path, "results.csv"),
                        header=0,
                        index_col=0,
                    )
                except Exception as e:
                    # print(f"Error is {e}\n")
                    if task_full not in dont_haves:
                        dont_haves[task_full] = 0
                    dont_haves[task_full] += 1
                    continue

                for task_faux_primary, results in all_results.groupby("task"):
                    if task_faux_primary.split("--")[0] != first_task:
                        continue  # for these exp we only care about the target task

                    try:
                        cur_metric = METRICS_OF_CHOICE[
                            task_faux_primary.split("_")[0].lower()
                        ]
                    except KeyError:  # halves data
                        cur_metric = METRICS_OF_CHOICE[
                            task_faux_primary[:-1].split("_")[0].lower()
                        ]
                    max_df = results[results["task"] == task_faux_primary][
                        results[cur_metric] == results[cur_metric].max()
                    ]
                    if max_df.shape[0] > 1:
                        # handle ties by taking the min checkpoint
                        idx_max = pd.to_numeric(
                            max_df["checkpoint"], errors="coerce"
                        ).idxmin()  # take the numeric checkpoint over the end one
                    else:
                        # find the best score since there are no ties
                        idx_max = results[cur_metric].idxmax()

                    best_result = results.loc[idx_max, :].to_numpy()
                    score = best_result[list(results.columns).index(cur_metric)]
                    cur_results = {
                        "primary": first_task,
                        "transfer": second_task,
                        "checkpoint": best_result[1],
                        "score": score,
                        "seed": seed_num,
                        "prop_main": prop_main,
                        "prop_support": prop_support,
                    }

                    results_list.append(cur_results)

    df = pd.DataFrame(results_list).sort_values("transfer")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, "validation_exp_results.csv"))

    all_seeds = set(["544460", "428404", "257803", "42", "801760"])

    for ((target_task_name, prop_main, prop_support), prop_df) in df.groupby(
        ["primary", "prop_main", "prop_support"]
    ):
        if len(prop_df) != 5:
            print(
                f"{target_task_name} at prop {prop_main} with support {prop_support} is missing seeds={all_seeds.difference(set(prop_df.seed.tolist()))}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="directory to find data ",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="directory to save data ",
        type=str,
    )
    args = parser.parse_args()
    gather_validation_exp(args)
    # example
    #  python3 ./bin/gather_validation_exp_results.py --data_dir <data_dir> --output_dir <output_dir>
