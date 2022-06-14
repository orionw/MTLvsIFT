import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.core.common import SettingWithCopyWarning

# due to a mistake in `create_differing_sizes` I divided by the number rather than multiply
# use this mapping to get back to a proportion
mapping_for_props = {1: 1, 0.33: 3, 0.5: 2, 2: 0.5, 3: 0.33}

# pesky Pandas Warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def plot_validation_exp(args: argparse.Namespace):
    df1 = pd.read_csv(args.data_dir1, index_col=0)
    df1["Method"] = "MTL"
    df2 = pd.read_csv(args.data_dir2, index_col=0)
    df2["Method"] = "STILTs"

    df = pd.concat([df1, df2], axis=0, sort=True)
    support_prop_name = "Support Task Proportion (relative to target at XXX)"

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    df = df[df["primary"].apply(lambda x: "SST" not in x)]
    df = df[df["prop_main"].apply(lambda x: x != 0.33)]  # remove for all three
    df["prop_support"] = df["prop_support"].apply(
        lambda x: mapping_for_props[x]
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(df["prop_main"].unique()),
        figsize=(5 * len(df["prop_main"].unique()), 5),
    )
    colors = ["#8a92ff", "#5cff59"]

    for index, ((target_task_name, prop_main), prop_df) in enumerate(
        df.groupby(["primary", "prop_main"])
    ):
        prop_df.prop_support = prop_df.prop_support.astype(str)
        sns.lineplot(
            data=prop_df,
            x="prop_support",
            y="score",
            hue="Method",
            ci=90,
            err_style="bars",
            ax=axes[index],
            palette=sns.color_palette(colors),
        )
        axes[index].set_xlabel(support_prop_name.replace("XXX", str(prop_main)))
        axes[index].set_ylabel("Score")
        axes[index].title.set_text(f"Target Task Proportion: {str(prop_main)}")
        if index != len(df["prop_main"].unique()) - 1:
            axes[index].legend([], [], frameon=False)
        # ax.set_xticks([0.33, 0.5, 1, 2, 3])

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"test_fig_qnli.png"), bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir1",
        help="directory to find the STILTs data ",
        type=str,
    )
    parser.add_argument(
        "--data_dir2",
        help="directory to find the MTL data ",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="directory to save data ",
        type=str,
    )
    args = parser.parse_args()
    plot_validation_exp(args)
    # example
    # python3 ./bin/plot_validation_exp.py --data_dir1 <data_dir1> --data_dir2 <data_dir2>  --output_dir <output_dir>
