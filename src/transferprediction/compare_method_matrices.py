import os
import glob
import argparse
import typing
import collections
import copy
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind_from_stats, linregress
from matplotlib.lines import Line2D

from transferprediction.settings import METRICS_OF_CHOICE, LOWER_TO_UPPER_TASKS

pd.set_option("display.float_format", lambda x: "%.4f" % x)


def compare_methods(args: argparse.Namespace):
    mtl = pd.read_csv(args.mtl_matrix)
    mtl_std_path = "/".join(
        args.mtl_matrix.split("/")[:-1]
        + [args.mtl_matrix.split("/")[-1].replace(".csv", "_std.csv")]
    )
    mtl_std = pd.read_csv(mtl_std_path)
    inter = pd.read_csv(args.inter_matrix, index_col=0)
    inter_std_path = "/".join(
        args.inter_matrix.split("/")[:-1]
        + [args.inter_matrix.split("/")[-1].replace("mean", "std")]
    )
    inter_std = pd.read_csv(inter_std_path, index_col=0)

    # standarize the format
    mtl.index = mtl.columns
    mtl_std.index = mtl.columns
    inter.columns = mtl.columns
    inter_std.columns = mtl.columns
    inter_std.index = mtl.columns

    diff_matrix = mtl - inter
    max_matrix = np.maximum(mtl.values, inter.values)
    arg_max_matrix = np.argmax((mtl.values, inter.values), 0)
    max_df_std = copy.deepcopy(mtl)
    for col_num in range(arg_max_matrix.shape[0]):
        for row_num in range(arg_max_matrix.shape[1]):
            max_df_std.iloc[row_num, col_num] = (
                mtl_std.iloc[row_num, col_num]
                if arg_max_matrix[row_num, col_num]
                else inter_std.iloc[row_num, col_num]
            )

    max_df = pd.DataFrame(max_matrix)
    max_df.columns = diff_matrix.columns
    max_df.index = diff_matrix.index

    mean_scores = pd.read_csv(
        "results_from_transfer/fixed-glue/single-glue-distil-not-halves/mean_single_pairs.csv",
        index_col=0,
    )
    std_scores = pd.read_csv(
        "results_from_transfer/fixed-glue/single-glue-distil-not-halves/std_single_pairs.csv",
        index_col=0,
    )

    mtl_all = pd.read_csv(
        "2021_test/mtl_all/mtl_all_best_results_size.csv",
        index_col=0,
    )
    mtl_all_dynamic = pd.read_csv(
        "2021_test/mtl_all/mtl_all_best_results_dynamic.csv",
        index_col=0,
    )
    mtl_all_uniform = pd.read_csv(
        "2021_test/mtl_all/mtl_all_best_results_uniform.csv",
        index_col=0,
    )

    results = copy.deepcopy(mtl)
    results = results.replace(results, 0)
    mask = copy.deepcopy(results)
    t_results = copy.deepcopy(results)
    size_results = copy.deepcopy(results)
    wrong = copy.deepcopy(results)
    size_matrix = copy.deepcopy(results)
    stat_results = copy.deepcopy(results)
    zero_mask = copy.deepcopy(results).astype(bool)
    mean_matrix = copy.deepcopy(results)
    std_matrix = copy.deepcopy(results)
    mtl_all_matrix = copy.deepcopy(results)

    with open("results_from_transfer/dataset_sizes.json", "r") as fin:
        sizes = json.load(fin)

    # import pdb; pdb.set_trace()
    for task_primary in mtl.index:
        for task_secondary in mtl.index:
            if task_primary == task_secondary:
                size_results[task_primary][task_secondary] = float("nan")
                results[task_primary][task_secondary] = float("nan")
                stat_results[task_primary][task_secondary] = float("nan")
                continue

            # make size matrix
            size_primary = sizes[task_primary]
            size_secondary = sizes[task_secondary]
            size_multiplier = max(
                size_primary / size_secondary, size_secondary / size_primary
            )
            # size_diff_greater = abs(size_primary - size_secondary) >= 5000
            if size_multiplier < args.size_threshold:
                size_results[task_primary][task_secondary] = 0
            elif size_primary > size_secondary:
                size_results[task_primary][task_secondary] = 1
                # mtl is higher when the primary size is smaller than the secondary size (since task_primary is actually task_Secondary)
            elif size_primary < size_secondary:
                size_results[task_primary][task_secondary] = -1

            # stat is negatie when the inter score is higher

            stat, p_val = ttest_ind_from_stats(
                mean1=mtl[task_primary][task_secondary],
                std1=mtl_std[task_primary][task_secondary],
                nobs1=5,
                mean2=inter[task_primary][task_secondary],
                std2=inter_std[task_primary][task_secondary],
                nobs2=5,
            )

            # Use with stat results
            stat_results[task_primary][task_secondary] = stat_results[
                task_primary
            ][task_secondary] = (
                mtl[task_primary][task_secondary]
                - inter[task_primary][task_secondary]
            )
            t_results[task_primary][task_secondary] = stat
            if p_val < 0.1:
                if task_secondary == "MRPC":
                    print(
                        f"{task_primary}, {task_secondary}, p, stat = {p_val}, {stat}"
                    )
                # print(f"{p_val} and {stat}")
                if stat < 0:
                    results[task_primary][
                        task_secondary
                    ] = -1  # MTL is lower, Inter is better
                    print(
                        f"{task_primary}, {task_secondary}, {mtl[task_primary][task_secondary]}, {inter[task_primary][task_secondary]}"
                    )
                else:
                    results[task_primary][
                        task_secondary
                    ] = 1  # MTL is higher, Inter is lower

    correct = 0
    total = 0
    zeros = 0
    mask_values = []
    for task_primary in mtl.index:
        for task_secondary in mtl.index:
            if task_primary == task_secondary:
                continue

            actual = results[task_primary][task_secondary]
            size_pred = size_results[task_primary][task_secondary]
            not_same = size_pred != actual

            if not_same:
                wrong[task_primary][task_secondary] = 1

            if actual not in [-1, 1]:
                continue

            if not actual:
                zeros += 1

            # if they have opposite values (size predicted opposite of actual) it's wrong
            # if we predict IFT is worse and it's equal, count as wrong
            acceptable_not_same = actual == 0 and size_pred == -1

            if not_same and (not acceptable_not_same):
                print(
                    f"Wrong on {task_primary}, {task_secondary} with sizes {sizes[task_primary]} vs {sizes[task_secondary]}, \
                         {max(sizes[task_secondary] / sizes[task_primary], sizes[task_primary] / sizes[task_secondary])}, score actual {actual} vs pred {size_pred}"
                )
                mask[task_primary][task_secondary] = 1
                mask_values.append(
                    "{0:.1f}".format(
                        float(stat_results[task_primary][task_secondary]) * 100
                    )
                )  # remove * 100 with t-stat
            else:
                correct += 1
            total += 1
    print(
        f"Size got {correct} out of {total} right, with {(size_results == 0).sum().sum()} zeros from size_preds and {zeros} from results"
    )
    print("Done!")

    fig_dims = (8, 5)
    fig, ax = plt.subplots()

    # Green Gray Blue
    colors = ["#5cff59", "#DCDCDC", "#8a92ff"]

    plotting_order = [
        "MNLI",
        "QQP",
        "QNLI",
        "SST-2",
        "CoLA",
        "STS-B",
        "MRPC",
        "RTE",
        "WNLI",
    ]
    reversed_ploting_order = list(reversed(plotting_order))
    mean_matrix = mean_matrix[plotting_order]
    results = results[plotting_order]
    wrong = wrong[plotting_order]
    mask = mask[plotting_order]
    size_matrix = size_matrix[plotting_order]
    t_results = t_results[plotting_order]
    stat_results = stat_results[plotting_order]
    max_df = max_df[plotting_order]
    max_df_std = max_df_std[plotting_order]
    std_matrix = std_matrix[plotting_order]

    mean_matrix = mean_matrix.reindex(reversed_ploting_order)
    std_matrix = std_matrix.reindex(reversed_ploting_order)
    results = results.reindex(reversed_ploting_order)
    wrong = wrong.reindex(reversed_ploting_order)
    mask = mask.reindex(reversed_ploting_order)
    size_matrix = size_matrix.reindex(reversed_ploting_order)
    t_results = t_results.reindex(reversed_ploting_order)
    stat_results = stat_results.reindex(reversed_ploting_order)
    max_df = max_df.reindex(reversed_ploting_order)
    max_df_std = max_df_std.reindex(reversed_ploting_order)

    sns.heatmap(
        stat_results * 100,
        center=0,
        annot=True,
        cbar=False,
        cmap=sns.color_palette(colors),
        fmt=".1f",
        ax=ax,
    )  # remove * 100 with t-stat
    sns.heatmap(
        results,
        center=0,
        annot=False,
        cbar=False,
        cmap=sns.color_palette(colors),
        ax=ax,
    )
    # ax = sns.heatmap(results, center=0, annot=False, cbar=False, cmap=sns.color_palette(colors),
    #                     mask=~mask.astype(bool), annot_kws={"color": "#ff3d3d", "weight": "bold"}, ax=ax)
    for text in ax.texts:
        text.set_size(11)
        text.set_color("black")
        if text.get_text() in mask_values:
            mask_values.remove(text.get_text())
            print(text.get_text)
            # HACK, because I know it isn't one but two have the same value at -2.7 so it flags them both
            if (text._x < 1 and text.get_text() != "-2.7") or text._y in [
                4.5,
                0.5,
            ]:
                continue
            text.set_color("#ff3d3d")  # Red
            text.set_weight("bold")

    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=4),
        Line2D([0], [0], color=colors[1], lw=4),
        Line2D([0], [0], color=colors[2], lw=4),
    ]

    ax.legend(
        custom_lines,
        [
            "Intermediate Fine Tuning is Better",
            "No Significant Difference",
            "Multi-Task Learning is Better",
        ],
        bbox_to_anchor=(0.75, 1.25),
    )

    plt.yticks(rotation=45)
    plt.tight_layout()
    ax.set_facecolor("#FFFAFA")
    ax.set_ylabel("Primary Task")
    ax.set_xlabel("Supporting Task")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(
        os.path.join(args.output_dir, "t-test.png"), bbox_inches="tight"
    )
    plt.close()
    results.to_csv(os.path.join(args.output_dir, "t-test.csv"))
    size_results.to_csv(os.path.join(args.output_dir, "size_results.csv"))

    name = copy.deepcopy(size_matrix)
    for col_name in mtl.index:
        for row_name in mtl.index:
            size_matrix[col_name][row_name] = sizes[row_name] / sizes[col_name]
            mean_matrix[col_name][row_name] = mean_scores.transpose()[
                row_name
            ].item()
            std_matrix[col_name][row_name] = std_scores.transpose()[
                row_name
            ].item()
            mtl_all_matrix[col_name][row_name] = mtl_all[
                mtl_all["task"] == row_name.lower()
            ]["score"].item()
            name[col_name][row_name] = row_name + "_" + col_name

    size_prop_name = "Log of Size Proportion (target / supporting)"
    wrong_name = "Prediction Was"
    t_score_name = "T-Score"
    stat_sig_name = "Statistical Significance"

    line_df = pd.DataFrame(
        {
            size_prop_name: np.log10(size_matrix.values.reshape(-1)),
            t_score_name: t_results.values.reshape(-1),
            stat_sig_name: results.fillna(0).astype(int).values.reshape(-1),
            wrong_name: ~(wrong.astype(bool).values.reshape(-1)),
            "name": name.values.reshape(-1),
        }
    )

    sns.scatterplot(
        data=line_df,
        x=size_prop_name,
        y=t_score_name,
        style=stat_sig_name,
        hue=wrong_name,
    )
    plt.axhline(y=0, color="r", linestyle="-")
    plt.savefig(
        os.path.join(args.output_dir, "t-test-scatter.png"), bbox_inches="tight"
    )
    plt.close()

    print(
        "LinReg with not sigs",
        linregress(line_df[size_prop_name], line_df[t_score_name]),
    )
    line_df_sig_only = line_df[line_df[stat_sig_name] != 0]
    line_df_only_non_sig = line_df[line_df[stat_sig_name] == 0]

    print(
        "LinReg with sigs only",
        linregress(
            line_df_sig_only[size_prop_name], line_df_sig_only[t_score_name]
        ),
    )

    change_single_minus_max = max_df - mean_matrix
    results_single = copy.deepcopy(change_single_minus_max)
    results_single = results_single.replace(results_single, 0)
    # change_single_minus_max = change_single_minus_max[
    #     list(reversed(change_single_minus_max.columns))
    # ]

    # Max vs Single-TasK Performance
    mask = (change_single_minus_max > 0).astype(float)
    for task_secondary in mask.columns:
        for task_primary in mask.columns:
            if task_primary == task_secondary:
                mask[task_secondary][task_primary] = float("nan")
                results_single[task_secondary][task_primary] = float("nan")

            else:
                stat, p_val = ttest_ind_from_stats(
                    mean1=max_df[task_primary][task_secondary],
                    std1=max_df_std[task_primary][task_secondary],
                    nobs1=5,
                    mean2=mean_matrix[task_primary][task_secondary],
                    std2=std_matrix[task_primary][task_secondary],
                    nobs2=5,
                )

                # t_results_single[task_primary][task_secondary] = stat
                if p_val < 0.1:
                    if stat < 0:
                        results_single[task_primary][
                            task_secondary
                        ] = -1  # MTL is lower, Inter is better
                    else:
                        results_single[task_primary][
                            task_secondary
                        ] = 1  # MTL is higher, Inter is lower

    plt.close()
    fig_dims = (8, 5)
    fig, ax = plt.subplots()
    sns.heatmap(
        change_single_minus_max * 100,
        center=0,
        annot=True,
        cbar=False,
        cmap=sns.color_palette(["#DCDCDC"]),
        fmt=".1f",
        ax=ax,
    )
    colors = sns.color_palette(["#FAA0A0", "#DCDCDC", "#FFFAA0"])
    # use mask for all scores and results_single for significant onces
    sns.heatmap(
        results_single, center=0, annot=False, cbar=False, cmap=colors, ax=ax
    )
    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=4),
        Line2D([0], [0], color=colors[1], lw=4),
        Line2D([0], [0], color=colors[2], lw=4),
    ]

    ax.legend(
        custom_lines,
        [
            "Using Single Fine-Tuning is Better",
            "No Statistical Difference",
            "Using Transfer Learning is Better",
        ],
        bbox_to_anchor=(0.75, 1.25),
    )
    plt.yticks(rotation=45)
    plt.tight_layout()
    ax.set_facecolor("#FFFAFA")
    ax.set_ylabel("Primary Task")
    ax.set_xlabel("Supporting Task")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(
        os.path.join(args.output_dir, "better_than_single.png"),
        bbox_inches="tight",
    )
    plt.close()

    change_mtl_all_vs_max = max_df - mtl_all_matrix
    change_mtl_all_vs_max = change_mtl_all_vs_max[
        list(reversed(change_mtl_all_vs_max.columns))
    ]
    print(f"MTL/STILTs vs MTL-ALL\n{change_mtl_all_vs_max}")

    best_max_df = max_df.mean(axis=1)
    best_mtl_all_df = mtl_all_matrix.max(axis=1)  # all the same value anyways
    pd.options.display.float_format = "{:,.3f}".format
    diff_values = best_max_df - best_mtl_all_df
    print(f"\nMTL/STILTs vs MTL-ALL Best only\n{diff_values.transpose()}")
    print(f"\nMTL-ALL STD\n{mtl_all}")

    print_out_values = lambda x, y: print(
        f"The values of {y} are {round(x.mean() * 100, 1)} & {' & '.join([str(round(n * 100, 1)) for n in list(reversed(x.values))])}"
    )
    print_out_values(best_max_df, "best max mean")
    print_out_values(best_mtl_all_df, "mtl_all")
    print_out_values(mtl_all_uniform.score, "mtl_all_uniform")
    print_out_values(mtl_all_dynamic.score, "mtl_all_dynamic")

    mtl_mean = mtl.mean(axis=1)
    inter.values[[np.arange(inter.shape[0])] * 2] = float(
        "nan"
    )  # make comparison fair
    inter_mean = inter.mean(axis=1)
    print_out_values(mtl_mean, "MTL")
    print_out_values(inter_mean, "STILTs")

    size_heuristic_ave_score = copy.deepcopy(size_results)
    for row_name in mtl.index:
        for col_name in mtl.index:
            size_pred = size_results[row_name][col_name]
            if size_pred == 1:
                size_heuristic_ave_score[row_name][col_name] = mtl[row_name][
                    col_name
                ]
                # use MTL
            elif size_pred == -1:
                size_heuristic_ave_score[row_name][col_name] = inter[row_name][
                    col_name
                ]
                # use IFT

    print_out_values(
        size_heuristic_ave_score.mean(axis=1), "Size Heuristic Average"
    )  # average in list from size heuristic
    print_out_values(max_df.max(axis=1), "best max max")  # best value in list

    return correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mtl_matrix",
        help="the directory to the results for a model",
        required=True,
    )
    parser.add_argument(
        "--inter_matrix",
        help="the directory to the results for the transfer learning / intermediate results",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="the directory to output the results",
        default=None,
    )
    parser.add_argument(
        "--size_threshold",
        help="the size of the threshold to use for a 'do-not-know' value",
        default=None,
        type=int,
    )
    args = parser.parse_args()
    print("Evaluating with args", args)
    if args.size_threshold is None:
        best_score = 0
        best_threshold = -1
        for i in range(20):
            print(f"Threshold {i}")
            args.size_threshold = i
            score = compare_methods(args)
            if score > best_score:
                best_score = score
                best_threshold = i
        print(
            f"Best score was {best_score} with a threshold of {best_threshold}"
        )
    else:
        compare_methods(args)
        # example
        # ./bin/compare_matrices --inter_matrix results_from_transfer/transfer_mean/mean_matrix.csv --mtl_matrix 2021_test/glue/score_matrix.csv --output_dir 2021_test/compare --size_threshold 1
