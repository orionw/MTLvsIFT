import os
import json
import re
import glob
from shutil import rmtree

import pandas as pd
import argparse

from transferprediction.settings import METRICS_OF_CHOICE


def remove_all_but_results(args: argparse.Namespace, folder: str):
    if not args.remove_all:
        raise Exception("Deleting all files when I shouldn't!")

    for file_path in glob.glob(os.path.join(args.file_dir, folder, "*")):
        file_name = file_path.split("/")[-1]
        if file_name in ["results.csv", "results.json", "loss_values.txt"]:
            continue
        else:
            os.remove(file_path)


def parse_glue_tasks(
    results: dict, task_name: str, secondary_task_name: str = None
):
    results_list = []
    for key_checkpoint_metric, checkpoint_results in results.items():
        try:
            extracted_name = re.search(
                f"^(.*)(_).*$", key_checkpoint_metric
            ).group(1)
            if extracted_name.lower() in task_name.lower():
                extracted_checkpoint = re.search(
                    f"^.*(_)(.*)$", key_checkpoint_metric
                ).group(2)
                results_list.append(
                    {
                        **{
                            "task": task_name
                            if len(task_name) < 10
                            else extracted_name,
                            "checkpoint": extracted_checkpoint,
                        },
                        **checkpoint_results,
                    }
                )
            # if there's two tasks, grab the second
            elif (
                secondary_task_name is not None
                and task_name != secondary_task_name
                and extracted_name.lower() in secondary_task_name.lower()
            ):
                extracted_checkpoint = re.search(
                    f"^.*(_)(.*)$", key_checkpoint_metric
                ).group(2)
                results_list.append(
                    {
                        **{
                            "task": secondary_task_name,
                            "checkpoint": extracted_checkpoint,
                        },
                        **checkpoint_results,
                    }
                )
        except Exception as e:
            print(e)
            print("CANNOT DELETE MODEL FILES, PLEASE CHECK!")
    return results_list


def parse_qa(results: dict, task_name: str):
    results_list = []
    for key_checkpoint_metric, checkpoint_results in results.items():
        try:
            extracted_name = re.search(
                f"^(.*)(_).*$", key_checkpoint_metric
            ).group(1)
            if extracted_name.lower() in ["best_f1"]:
                extracted_checkpoint = re.search(
                    f"^.*(_)(.*)$", key_checkpoint_metric
                ).group(2)
                results_list.append(
                    {
                        "task": task_name,
                        "checkpoint": extracted_checkpoint,
                        extracted_name: checkpoint_results,
                    }
                )
        except Exception as e:
            print(e)
            print("CANNOT DELETE MODEL FILES, PLEASE CHECK!")
    return results_list


def save_and_remove_others(args):
    for folder in os.listdir(args.file_dir):

        if not os.path.isdir(os.path.join(args.file_dir, folder)):
            # dont need to check files
            continue

        real_task = folder.split("--")[0] if not args.single else folder
        secondary_task = folder.split("--")[1] if not args.single else None
        if args.real_task is not None:
            real_task = args.real_task

        # if not os.path.isfile(os.path.join(args.file_dir, folder, "results.json")):
        #     continue

        results_file = os.path.join(args.file_dir, folder, "results.json")
        print("Results file", results_file)
        if (
            len(glob.glob(os.path.join(args.file_dir, folder, "checkpoint-*")))
            < 2
            and os.path.isfile(
                os.path.join(results_file.replace("json", "csv"))
            )
            and not args.redo
        ):
            if args.remove_all:
                remove_all_but_results(
                    args, folder
                )  # just to be sure it's gone
            continue  # we've already cleared this file

        # load each data folders results info
        print(
            "Checking folder",
            folder,
            "inside folders:",
            len(glob.glob(os.path.join(args.file_dir, folder, "checkpoint-*"))),
        )
        with open(results_file, "r") as fin:
            for index, line in enumerate(fin):
                assert index == 0, "index was more than 0!"
                results = json.loads(line)

        # parse the results info into a dataframe
        results_list = (
            parse_glue_tasks(results, real_task, secondary_task)
            if "multiqa" not in args.file_dir
            else parse_qa(results, real_task)
        )

        # if it's not empty, remove extra folders
        results_df = pd.DataFrame(results_list)
        if not results_df.empty:
            print(results_df.head())
            try:
                chosen_metric = METRICS_OF_CHOICE[
                    real_task.split("_")[0].lower()
                ]
            except Exception as e:
                chosen_metric = METRICS_OF_CHOICE[
                    real_task.split("_")[0][:-1].lower()
                ]  # attempt to see if this was halves data
            best_checkpoint = results_df["checkpoint"][
                results_df[results_df["task"] == real_task][
                    chosen_metric
                ].idxmax()
            ]
            print(
                "Best checkpoint for {} was {}".format(folder, best_checkpoint)
            )

            for checkpoint in results_df["checkpoint"].unique():
                if (
                    checkpoint != best_checkpoint or args.remove_all
                ) and checkpoint.isnumeric():
                    print("Removing checkpoint-{}".format(checkpoint))
                    try:
                        rmtree(
                            os.path.join(
                                args.file_dir,
                                folder,
                                "checkpoint-{}".format(checkpoint),
                            )
                        )
                    except Exception as e:
                        print(
                            "Could not delete folder {}".format(
                                os.path.join(
                                    args.file_dir,
                                    folder,
                                    "checkpoint-{} with error {}".format(
                                        checkpoint, e
                                    ),
                                )
                            )
                        )
            try:
                if (
                    best_checkpoint.lower() not in folder.lower()
                    or args.remove_all
                ):  # if checkpoint is best, not main dir model
                    print(
                        "Removing pytorch model in root dir at",
                        os.path.join(
                            args.file_dir, folder, "pytorch_model.bin"
                        ),
                    )
                    os.remove(
                        os.path.join(args.file_dir, folder, "pytorch_model.bin")
                    )
            except Exception as e:
                print(
                    "Could not delete folder {}".format(
                        os.path.join(args.file_dir, folder, "pytorch_model.bin")
                    )
                )
            results_df = results_df.sort_values("checkpoint")
            results_df.to_csv(
                os.path.join(args.file_dir, folder, "results.csv")
            )
            if args.remove_all:
                remove_all_but_results(args, folder)

        else:
            import pdb

            pdb.set_trace()
            raise Exception("Empty DF with checkpoints to clear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir")
    parser.add_argument("--single", default=False, action="store_true")
    parser.add_argument("--remove_all", default=False, action="store_true")
    parser.add_argument("--real_task", default=None, type=str)
    parser.add_argument("--redo", default=False, action="store_true")
    args = parser.parse_args()
    try:
        save_and_remove_others(args)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)  # we don't want to kill the flow for this
