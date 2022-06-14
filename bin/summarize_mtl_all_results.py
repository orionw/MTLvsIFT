import os
import glob
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from transferprediction.settings import METRICS_OF_CHOICE, LOWER_TO_UPPER_TASKS
from transferprediction.find_save_best_only import parse_glue_tasks


def summarize_mtl_all(args: argparse.Namespace):
    results_list = []
    dont_haves = {}
    all_tasks = "CoLA SST-2 MRPC QQP STS-B MNLI QNLI RTE WNLI"
    for mtl_type in ["", "size", "uniform"]:
        for task_run_path in glob.glob(
            os.path.join(
                args.data_dir + f"-{mtl_type}" if mtl_type else args.data_dir,
                "*",
            )
        ):  # task path is also seed path
            seed_num = task_run_path.split("/")[-1]
            task = f"mtl_all_{mtl_type}"
            try:
                with open(
                    os.path.join(task_run_path, "results.json"), "r"
                ) as fin:
                    all_results_json = json.load(fin)
                parsed_results = parse_glue_tasks(
                    all_results_json, all_tasks, mtl_type
                )
                all_results = pd.DataFrame(parsed_results)
            except Exception as e:
                # print(f"Error is {e}\n")
                # print(f"No data for {os.path.join(task_run_path, 'results.csv')}")
                if task_run_path not in dont_haves:
                    dont_haves[task] = 0
                dont_haves[task] += 1
                continue

            for task_faux_primary, results in all_results.groupby("task"):
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
                    "mtl_type": mtl_type,
                    "checkpoint": best_result[1],
                    "score": score,
                    "seed": seed_num,
                    "task": task_faux_primary,
                }

                results_list.append(cur_results)

    df = pd.DataFrame(results_list).sort_values("mtl_type")
    df.loc[df["mtl_type"] == "", "mtl_type"] = "dynamic"

    for name, count in dont_haves.items():
        print(f"Don't have {count} {name}s")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    df.to_csv(os.path.join(args.output_dir, "mtl_all_results.csv"))

    all_seeds = set(["544460", "428404", "257803", "42", "801760"])

    for ((mtl_type, task_name), prop_df) in df.groupby(["mtl_type", "task"]):
        if len(prop_df) != 5:
            print(
                f"{mtl_type} at {task} is missing seeds={all_seeds.difference(set(prop_df.seed.tolist()))}"
            )

    print(df.groupby(["mtl_type"], as_index=False).agg("mean"))
    print(df.groupby(["mtl_type", "task"], as_index=False).agg("mean"))

    for save_sampling_name in ["size", "dynamic", "uniform"]:
        best_results = []
        for ((task_name), cur_df) in df[
            df.mtl_type == save_sampling_name
        ].groupby(["task"]):
            best_results.append(
                {
                    "task": task_name,
                    "score": cur_df.score.mean(),
                    "std": cur_df.score.std(),
                }
            )

        best_df = pd.DataFrame(best_results)
        print(f"Best results are\n {best_df}")
        best_df.to_csv(
            os.path.join(
                args.output_dir,
                f"mtl_all_best_results_{save_sampling_name}.csv",
            )
        )

    ax = sns.boxplot(
        x="task", y="score", hue="mtl_type", data=df, palette="Set3"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"mtl_all_comp.png"), bbox_inches="tight"
    )
    plt.close()

    print("Done!")


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
    summarize_mtl_all(args)
    # example
    #  python3 ./bin/summarize_mtl_all_results.py --data_dir <DATA_DIR> --output_dir <OUTPUT_DIR>
