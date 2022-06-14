import os
import glob
import argparse
import typing
import collections
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transferprediction.settings import METRICS_OF_CHOICE, LOWER_TO_UPPER_TASKS

pd.set_option("display.float_format", lambda x: "%.4f" % x)


def get_dataset_paths(
    base_dir: str, use_seed: bool = False
) -> typing.List[typing.Tuple]:
    if use_seed:
        all_dataset_paths = []
        for seed_path in glob.glob(os.path.join(base_dir, "*")):
            seed_num = seed_path.split("/")[-1]
            tuple_of_paths = [
                (seed_num, path)
                for path in glob.glob(os.path.join(seed_path, "*"))
            ]
            all_dataset_paths.extend(tuple_of_paths)
        return all_dataset_paths
    else:
        seed_num = None
        tuple_of_paths = [
            (seed_num, path) for path in glob.glob(os.path.join(base_dir, "*"))
        ]
        return tuple_of_paths


def create_single_matrix(args: argparse.Namespace):
    if args.output_dir is None:
        args.output_dir = "results/single"
    data_list = []
    if args.use_seed is False:
        raise NotImplementedError(
            "Should always expect multiple seeds for single matrix creation"
        )

    all_data_paths = get_dataset_paths(args.dir_path, args.use_seed)
    assert all_data_paths, "no paths found!"
    for (seed, dataset_path) in all_data_paths:
        task = dataset_path.split("/")[-1]
        print(dataset_path)
        try:
            results = pd.read_csv(
                os.path.join(dataset_path, "results.csv"),
                header=0,
                index_col=0,
            )
            # we are sorting by strings
            results["checkpoint"] = results["checkpoint"].apply(
                lambda x: str(x)
            )
            results = results.sort_values("checkpoint")
        except Exception as e:
            print(e)
            continue

        try:
            cur_metric = METRICS_OF_CHOICE[task.lower()]
            halves = False
        except KeyError:
            cur_metric = METRICS_OF_CHOICE[task[:-1].lower()]
            halves = True

        max_df = results[results[cur_metric] == results[cur_metric].max()]
        if max_df.shape[0] > 1:
            # handle ties by taking the min checkpoint
            idx_max = pd.to_numeric(
                max_df["checkpoint"], errors="coerce"
            ).idxmin()  # take the numeric checkpoint over the end one
        else:
            # find the best score since there are no ties
            idx_max = results[cur_metric].idxmax()

        best_result = results.iloc[idx_max, :].to_numpy()
        score = best_result[list(results.columns).index(cur_metric)]
        cur_results = {
            "task": best_result[0],
            "checkpoint": best_result[1],
            "score": score,
        }
        if args.use_seed:
            cur_results["seed"] = seed

        data_list.append(cur_results)

    results = pd.DataFrame(data_list)
    if results.empty:
        raise Exception("Empty DataFrame, have you double checked the path?")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print("Saving single results to {}".format(args.output_dir))
    results.to_csv(
        os.path.join(args.output_dir, "all_single_pairs.csv"), index=None
    )
    min_df = results.loc[results.groupby("task")["score"].idxmin()]
    min_df.to_csv(os.path.join(args.output_dir, "min_single_pairs.csv"))
    max_df = results.loc[results.groupby("task")["score"].idxmax()]
    max_df.to_csv(os.path.join(args.output_dir, "max_single_pairs.csv"))
    mean_results = results.groupby("task")["score"].mean()
    mean_results.to_csv(
        os.path.join(args.output_dir, "mean_single_pairs.csv"), header=True
    )

    # work on median
    median_results = results.groupby("task")["score"].median()
    median_results.to_csv(
        os.path.join(args.output_dir, "median_single_pairs.csv"),
        header=True,
    )
    median_scores = median_results.to_list()
    try:
        # if we can get the median, process and output them
        median_df = pd.concat(
            [
                group[group.score.isin([group.score.median()])].iloc[[0]]
                for (name, group) in results.groupby("task")
            ],
            axis=0,
        ).reset_index(drop=True)
        median_df.to_csv(
            os.path.join(args.output_dir, "median_single_pairs_all.csv"),
            header=True,
        )
        assert median_df.shape[0] == len(
            results.task.unique()
        ), "did not find enough medians, error!"
        median_closest = (
            median_df["seed"]
            + "/"
            + median_df["task"]
            + "/checkpoint-"
            + median_df["checkpoint"].astype(str)
        )
        with open(
            os.path.join(args.output_dir, "probe_list_median.txt"), "w"
        ) as fout:
            # we can use this to copy and paste into the next script
            fout.write(" ".join(median_closest.to_list()))
    except IndexError:
        pass

    # get all the standard deviations
    results.groupby("task")["score"].std().to_csv(
        os.path.join(args.output_dir, "std_single_pairs.csv"), header=True
    )
    probes_to_gather = []
    for task, task_df in results.groupby("task"):
        mean_task_score = mean_results[task]
        score_close = abs(task_df["score"] - mean_task_score)
        closest_to_mean = task_df.loc[score_close.idxmin()]
        # NOTE: this could potentially break if the checkpoint is a string and thus we don't need this
        checkpoint_info = "checkpoint-" + str(closest_to_mean["checkpoint"])
        if halves:
            upper_task = LOWER_TO_UPPER_TASKS[task[:-1].lower()] + task[-1]
        else:
            upper_task = LOWER_TO_UPPER_TASKS[task.lower()]
        closest_to_mean["path"] = os.path.join(
            closest_to_mean["seed"],
            upper_task,
            checkpoint_info,
        )
        probes_to_gather.append(closest_to_mean.to_dict())

    closest_df = pd.DataFrame(probes_to_gather)
    closest_df.to_csv(os.path.join(args.output_dir, "closest_probes_mean.csv"))
    with open(
        os.path.join(args.output_dir, "probe_list_mean.txt"), "w"
    ) as fout:
        # we can use this to copy and paste into the next script
        fout.write(" ".join(closest_df["path"].to_list()))


def compared_with_single(args, results, primary_dict, primary_std_dict):
    validation = []
    for index, (row) in results.iterrows():
        task = row["primary"][:-1]
        # only match the pairs of the same task
        if task != row["transfer"][:-1]:
            continue

        score_alone = primary_dict[task]
        score_alone_std = primary_std_dict[task]

        score_pairs = row["score"]
        score_pairs_std = row["std"]

        std_away_from_alone = (
            (score_pairs - score_alone) / score_alone_std
            if score_alone_std != 0.0
            else 0
        )
        std_away_from_pairs = (
            (score_pairs - score_alone) / score_pairs_std
            if score_pairs_std != 0.0
            else 0
        )

        validation.append(
            {
                "primary": row["primary"],
                "std_away_from_alone": std_away_from_alone,
                "std_away_from_pairs": std_away_from_pairs,
                "pairs_score": score_pairs,
                "alone_score": score_alone,
                "std_pairs": score_pairs_std,
                "std_alone": score_alone_std,
            }
        )
    val_df = pd.DataFrame(validation)
    print("The validation DF is\n", val_df)
    val_df.to_csv(os.path.join(args.output_dir, "validation.csv"))


def get_primary_dicts_from_path(
    args: argparse.Namespace, single_data_path: str
):
    full_primary = pd.read_csv(single_data_path, index_col=None, header=0)
    primary = full_primary[["task", "score"]]  # only need task and score
    assert primary.shape[1] == 2, "got more values than expected: {}".format(
        primary.columns
    )
    primary_dict = {array[0]: array[1] for array in list(primary.to_numpy())}

    primary_std = pd.read_csv(
        single_data_path.replace(args.single_data_type, "std"),
        index_col=None,
        header=0,
    )
    primary_std = primary_std[["task", "score"]]  # only need task and score
    assert (
        primary_std.shape[1] == 2
    ), "got more values than expected: {}".format(primary_std.columns)
    primary_std_dict = {
        array[0]: array[1] for array in list(primary_std.to_numpy())
    }
    return primary_dict, primary_std_dict


def validate_both_halves_of_matrix(diff_dict, results):
    differences_std = {}
    for key, values in diff_dict.items():
        primary, transfer = key.split("--")
        # get the results of the 5 seeds for each pair combination
        ave = results[
            (results["primary"] == primary) & (results["transfer"] == transfer)
        ]
        # how much is the difference compared to the 5 seed run difference
        differences_std[key] = (
            pd.Series(values).median() / ave["std"].iloc[0],
            pd.Series(values).mean() / ave["std"].iloc[0],
        )

    for key, (median_val, mean_val) in differences_std.items():
        if abs(median_val) > 1.0 and abs(mean_val) > 1.0:
            print(
                "The tasks, {} were {} (median) and {} (mean) std away from the 5 seed average".format(
                    key, median_val, mean_val
                )
            )


def create_multi_matrix(args):
    if args.output_dir is None:
        args.output_dir = "results/multi"
    single_data_path = args.single_data_path + "{}_single_pairs.csv".format(
        args.single_data_type
    )
    data_list = []
    all_data_paths = get_dataset_paths(args.dir_path, args.use_seed)
    for (seed, dataset_path) in all_data_paths:
        task = dataset_path.split("/")[-1]
        first_task, second_task = task.split("--")
        try:
            all_results = pd.read_csv(
                os.path.join(dataset_path, "results.csv"), header=0, index_col=0
            )
        except Exception as e:
            print(e)
            continue

        for task_faux_primary, results in all_results.groupby("task"):
            try:
                cur_metric = METRICS_OF_CHOICE[task_faux_primary.lower()]
            except KeyError:  # halves data
                cur_metric = METRICS_OF_CHOICE[task_faux_primary[:-1].lower()]
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
                "primary": task_faux_primary,
                "transfer": first_task
                if first_task.lower() != task_faux_primary.lower()
                else second_task,
                "checkpoint": best_result[1],
                "score": score,
            }
            if args.use_seed:
                cur_results["seed_num"] = seed
            data_list.append(cur_results)

    # read in primary forms
    primary_dict, primary_std_dict = get_primary_dicts_from_path(
        args, single_data_path
    )

    def lookup_subtract(row, key="score"):
        return (row[key] - primary_dict[row["primary"]]) / primary_dict[
            row["primary"]
        ]

    def lookup_prop(row, key="std"):
        return row[key] / primary_dict[row["primary"]]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    print("Saving multi results to {}".format(args.output_dir))

    results = pd.DataFrame(data_list)
    if results.empty:
        raise Exception("DF is empty, perhaps wrong directory?")

    if "seed_num" in results.columns:  # pylint: disable=E1135
        list_of_dicts = []
        diff_dict = {}
        for ((primary_task, transfer_task), df_by_pair) in results.groupby(
            ["primary", "transfer"]
        ):
            # if we have multiple sets, validate that both parts of the triangular matrix are equal
            if len(set(df_by_pair.seed_num.tolist())) * 2 == len(
                df_by_pair.seed_num.tolist()
            ):
                for seed_num, seed_pair_df in df_by_pair.groupby("seed_num"):
                    assert (
                        len(seed_pair_df) == 2
                    ), "got too many values for both halves of the matrix"
                    primary_for_seed = seed_pair_df.primary.unique().tolist()
                    assert (
                        len(primary_for_seed) == 1
                    ), "got too many in list for primary seed"
                    transfer_for_seed = seed_pair_df.transfer.unique().tolist()
                    assert (
                        len(transfer_for_seed) == 1
                    ), "got too many in list for transfer seed"
                    name_key = primary_for_seed[0] + "--" + transfer_for_seed[0]
                    if name_key not in diff_dict:
                        diff_dict[name_key] = []
                    diff_dict[name_key].append(
                        seed_pair_df.score.iloc[0] - seed_pair_df.score.iloc[1]
                    )

            average = df_by_pair["score"]
            std_for_pair = average.std()
            if args.aggregate == "mean":
                average = average.mean()
            elif args.aggregate == "max":
                average = average.max()
            elif args.aggregate == "min":
                average = average.min()
            elif args.aggregate == "median":
                average = average.median()
            else:
                raise Exception(
                    "Cannot parse aggregate type {}".format(args.aggregate)
                )
            list_of_dicts.append(
                {
                    "primary": primary_task,
                    "transfer": transfer_task,
                    "score": average,
                    "std": std_for_pair,
                }
            )
        results = pd.DataFrame(list_of_dicts)

    results.to_csv(
        os.path.join(args.output_dir, "all_pairs.csv"), index=None
    )  # values from pair experiments

    # how do these results compare with the single, in terms of validation
    if args.single_full_data_path is not None:
        val_primary_dict, val_primary_std_dict = get_primary_dicts_from_path(
            args, args.single_full_data_path
        )
        compared_with_single(
            args, results, val_primary_dict, val_primary_std_dict
        )
        if len(diff_dict):
            validate_both_halves_of_matrix(diff_dict, results)

    # since we may be grabbing doubles, make sure there are no duplicates
    results = results.drop_duplicates()
    keeps = []
    for info, group in results.groupby(["primary", "transfer"]):
        if len(group) == 1:
            keeps.append(group)
        else:
            best = group.sort_values(
                ["score", "checkpoint"], ascending=[False, True]
            )
            keeps.append(best.iloc[[0]])

    results = pd.concat(keeps, axis=0)
    results_matrix = results.pivot(
        index="primary", columns="transfer", values="score"
    )
    results_matrix.to_csv(
        os.path.join(args.output_dir, "score_matrix.csv"), index=None
    )

    # get difference in results absolute
    diff_results = copy.deepcopy(results)
    diff_results["score"] = results.apply(
        lambda row: row["score"] - primary_dict[row["primary"]], axis=1
    )
    diff_matrix_absolute = diff_results.pivot(
        index="primary", columns="transfer", values="score"
    )
    diff_matrix_absolute.to_csv(
        os.path.join(args.output_dir, "diff_score_matrix.csv"), index=None
    )

    results["score"] = results.apply(
        lookup_subtract, axis=1
    )  # values proportional to primary experiments
    if args.use_seed:
        # get regular std
        results_matrix = results.pivot(
            index="primary", columns="transfer", values="std"
        )
        results_matrix.to_csv(
            os.path.join(args.output_dir, "score_matrix_std.csv"), index=None
        )

        results["std"] = results.apply(
            lookup_prop, args=("std",), axis=1
        )  # values proportional to primary experiments
    results["primary"] = results["primary"].apply(lambda x: x.upper())
    matrix = results.pivot(index="primary", columns="transfer", values="score")
    matrix.columns = [item.upper() for item in matrix.columns]
    matrix.to_csv(os.path.join(args.output_dir, "matrix_prop_diff.csv"))
    if args.use_seed:
        matrix = results.pivot(
            index="primary", columns="transfer", values="std"
        )
        matrix.columns = [item.upper() for item in matrix.columns]
        matrix.to_csv(os.path.join(args.output_dir, "matrix_prop_std.csv"))

    if not args.dont_validate:
        print("### Validating data... ###")
        seed_runs = set(
            collections.Counter(
                pd.DataFrame(data_list).seed_num.tolist()
            ).values()
        )
        if len(seed_runs) != 1:
            print(
                "did not all get the same number of runs (should finish running pairs): {}".format(
                    seed_runs
                )
            )
        score_nums = set(
            collections.Counter(pd.DataFrame(data_list).score.tolist()).values()
        )
        if len(score_nums) != 1:
            print(
                "Got results where the exact same score was acheived (not an error, something to watch out for): {}".format(
                    score_nums
                )
            )


def create_transfer_matrix(args):
    primary_dict, primary_std_dict = get_primary_dicts_from_path(
        args, args.single_data_path
    )

    mapping = {}
    for path_with_results in glob.glob(os.path.join(args.dir_path, "*")):
        model_name = path_with_results.split("/")[-1]
        values = model_name.split("-")
        seed, task_init = values[0], values[1]

        for metric in ["max", "mean", "median", "min", "std"]:
            if metric not in mapping:
                mapping[metric] = []
            results = pd.read_csv(
                os.path.join(
                    path_with_results, "{}_single_pairs.csv".format(metric)
                ),
                index_col=None,
            )
            for (task, score), group_df in results.groupby(["task", "score"]):
                mapping[metric].append(
                    {
                        "initial": task_init,
                        "secondary": task,
                        "score": score,
                        "diff": score - primary_dict[task],
                    }
                )

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    for metric in mapping.keys():
        cur_df = pd.DataFrame(mapping[metric])
        matrix_df = cur_df.pivot(
            index="secondary", columns="initial", values="score"
        )
        cur_df.to_csv(
            os.path.join(args.output_dir, "{}_results.csv".format(metric))
        )
        matrix_df.to_csv(
            os.path.join(args.output_dir, "{}_matrix.csv".format(metric))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path",
        help="the directory to the results for a model",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="the directory to output the results",
        default=None,
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="whether this is a single run or not",
        default=False,
    )
    parser.add_argument(
        "--single_data_type",
        help="the type of aggregation file to use for single data from the single-matrix data",
        default="mean",
    )
    parser.add_argument(
        "--single_data_path",
        help="the base path to the single data aggregation file",
        default="results/single/",
    )
    parser.add_argument(
        "--single_full_data_path",
        help="the base path the single results from the full datasets, if using halves",
        default=None,
    )
    parser.add_argument(
        "--aggregate",
        help="whether to aggregate the pairs of run seeds by min/mean/max for the pairs data",
        default="mean",
    )
    parser.add_argument(
        "--use_seed",
        help="whether this is a single run or should be aggregated over multiple seeds",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dont_validate",
        help="whether to run some simple analysis on the data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--transfer",
        help="whether to create the matrix from the transfer data",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    print("Evaluating with args", args)
    if args.single is True:
        create_single_matrix(args)
    elif args.transfer:
        create_transfer_matrix(args)
    else:
        create_multi_matrix(args)
