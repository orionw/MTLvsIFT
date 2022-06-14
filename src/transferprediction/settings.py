import os

# constants

GLUE_TASKS = [
    "cola",
    "sst-2",
    "mrpc",
    "qqp",
    "sts-b",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "none",  # used for base model no fine-tuning
]


METRICS_OF_CHOICE = {
    "cola": "mcc",
    "sst-2": "acc",
    "mrpc": "acc_and_f1",
    "qqp": "acc_and_f1",
    "sts-b": "spearmanr",
    "mnli": "acc",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}

METRICS_FOR_TASK = {
    "cola": ["mcc"],
    "sst-2": ["acc"],
    "mrpc": ["acc", "f1", "acc_and_f1"],
    "qqp": ["acc", "f1", "acc_and_f1"],
    "sts-b": ["corr", "pearson", "spearmanr"],
    "mnli": ["acc"],
    "qnli": ["acc"],
    "rte": ["acc"],
    "wnli": ["acc"],
}

LOWER_TO_UPPER_TASKS = {
    "cola": "CoLA",
    "sst-2": "SST-2",
    "mrpc": "MRPC",
    "qqp": "QQP",
    "sts-b": "STS-B",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
}

# logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""

ALL_TASKS = GLUE_TASKS
