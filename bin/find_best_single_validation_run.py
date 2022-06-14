import os
import glob
import pandas as pd
import numpy as np
import argparse


def find_best_singles(args: argparse.Namespace):
    results_list = []
    dont_haves = {}
    for seed_path in glob.glob(os.path.join(args.data_dir, "*")):
        seed_num = seed_path.split("/")[-1]
        for prop_path in glob.glob(os.path.join(seed_path, "*")):
            prop_main = prop_path.split("/")[-1]
            for task_run_path in glob.glob(os.path.join(prop_path, "*")):
                task_full_name = task_run_path.split("/")[-1]
                prop_support = task_run_path.split("/")[-1].split("_")[-1]
                support_task_name = task_run_path.split("/")[-1].split("_")[0]

                if not os.path.isfile(
                    os.path.join(task_run_path, "results.csv")
                ):
                    print(
                        f"No data for {os.path.join(task_run_path, 'results.csv')}"
                    )
                    if task_full_name not in dont_haves:
                        dont_haves[task_full_name] = 0
                    dont_haves[task_full_name] += 1
                    continue

                results_data = pd.read_csv(
                    os.path.join(task_run_path, "results.csv"), index_col=0
                )
                checkpoints_left = list(
                    glob.glob(os.path.join(task_run_path, "checkpoint-*"))
                )
                if not len(checkpoints_left):
                    import pdb

                    pdb.set_trace()
                else:
                    best_checkpoint_name = checkpoints_left[0].split("-")[-1]
                best_score = results_data[
                    results_data["checkpoint"] == int(best_checkpoint_name)
                ][results_data.columns[-1]].iloc[0]
                results_list.append(
                    {
                        "seed": seed_num,
                        "prop_main": prop_main,
                        "prop_support": prop_support,
                        "score": best_score,
                        "support_task_name": support_task_name,
                        "path_to_model": "/".join(
                            os.path.join(
                                task_run_path, checkpoints_left[0]
                            ).split("/")[-4:]
                        ),
                        "target_data": (
                            "SST-2" if support_task_name == "QQP" else "QNLI"
                        )
                        + f"_{prop_main}",
                    }
                )

    df = pd.DataFrame(results_list)
    metric_to_use = np.max if args.metric == "max" else np.median

    all_seeds = set(["544460", "428404", "257803", "42", "801760"])
    print(dont_haves)

    best_list = []
    # go through each setting and find the best seed for the given trio
    for ((target_task_name, prop_main, prop_support), prop_df) in df.groupby(
        ["support_task_name", "prop_main", "prop_support"]
    ):

        if len(prop_df) != 5 and len(set(prop_df.seed.unique())) != 5:
            print(
                f"{target_task_name} at prop {prop_main} with support {prop_support} is missing seeds={all_seeds.difference(set(prop_df.seed.tolist()))}"
            )
        else:
            best_current = prop_df[
                prop_df.score == metric_to_use(prop_df.score)
            ]
            if len(best_current) > 1:
                best_list.append(best_current.iloc[[0]])
            else:
                best_list.append(best_current)

    best_df = pd.concat(best_list, axis=0)
    for ((target_data), prop_df) in best_df.groupby(["target_data"]):
        print(f"\n{target_data}:")
        for index, (row) in prop_df.iterrows():
            print(row["path_to_model"])

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    best_df.to_csv(os.path.join(args.output_dir, "best_single_paths.csv"))


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
    parser.add_argument(
        "--metric",
        help="which metric to use to get the best path",
        type=str,
        choices=["max", "median"],
        default="max",
    )
    args = parser.parse_args()
    find_best_singles(args)

    # python3 ./bin/find_best_single_validation_run.py --data_dir <DATA_DIR> --output_dir <PATH>
