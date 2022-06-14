import os
import json
import csv
import pandas as pd

GLUE_IO = {
    "cola": {
        "header": None,
        "index_col": None,
        "label": 1,
        "text": [3],
    },
    "sst-2": {
        "header": 0,
        "index_col": 0,
        "label": "label",
        "text": ["sentence"],
    },
    "mrpc": {
        "header": 0,
        "index_col": 0,
        "label": "\ufeffQuality",
        "text": ["#1 String", "#2 String"],
    },
    "qqp": {
        "header": 0,
        "index_col": 0,
        "label": "is_duplicate",
        "text": ["question1", "question2"],
    },
    "sts-b": {
        "header": 0,
        "index_col": 0,
        "label": "score",
        "text": ["sentence1", "sentence2"],
    },
    "mnli": {
        "header": 0,
        "index_col": 0,
        "label": "gold_label",
        "text": ["sentence1", "sentence2"],
    },
    "qnli": {
        "header": 0,
        "index_col": 0,
        "label": "label",
        "text": ["question", "sentence"],
    },
    "rte": {
        "header": 0,
        "index_col": 0,
        "label": "label",
        "text": ["sentence1", "sentence2"],
    },
    "wnli": {
        "header": 0,
        "index_col": 0,
        "label": "label",
        "text": ["sentence1", "sentence2"],
    },
}

ALL_DEFINITIONS = [GLUE_IO]


def load_glue(folder_name: str, data_folder: str, split: str):
    """
    FolderName: the task name
    data_folder: the folder containing the data, root dir
    split: train, dev, test
    """
    # deal with the weirdness of glue
    if folder_name not in ["CoLA"]:
        df = pd.read_csv(
            os.path.join(data_folder, folder_name, split + ".tsv"),
            quoting=csv.QUOTE_NONE,
            sep=None,  # figure it out
            escapechar="|",  # so that nothing is escaped
            index_col=None,
            header=0,
            engine="python",
        )
    else:
        df = pd.read_csv(
            os.path.join(data_folder, folder_name, split + ".tsv"),
            header=None,
            sep="\n",
        )
        df = df[0].str.split("\t", expand=True)

    if folder_name == "QQP":
        df = df.dropna()

    return df
