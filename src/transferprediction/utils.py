import contextlib
import logging
import typing
from transformers.data.processors.glue import *
from transformers.data.metrics import *

from transferprediction import settings


flatten = lambda l: [item for sublist in l for item in sublist]


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["rte"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}


def configure_logging(verbose: bool = False) -> logging.Handler:
    """Configure logging and return the log handler.

    This function is useful in scripts when logging should be set up
    with a basic configuration.

    Parameters
    ----------
    verbose : bool, optional (default=False)
        If ``True`` set the log level to DEBUG, else set it to INFO.

    Returns
    -------
    logging.Handler
        The log handler set up by this function.
    """
    # unset the log level from root (defaults to WARNING)
    logging.root.setLevel(logging.NOTSET)

    # set up the log handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

    # attach the log handler to root
    logging.root.addHandler(handler)

    return handler


class FileLogging(contextlib.AbstractContextManager):
    """A context manager for logging to a file.

    Use this context manager to log output to a file. The context manager
    returns the log handler for the file. The manager attaches the handler to
    the root logger on entering the context and detaches it upon exit.

    Parameters
    ----------
    file_path : str, required
        The path at which to write the log file.

    Example
    -------
    Use the context manager as follows::

        with FileLogging('foo.log') as log_handler:
            # modify the log hander if desired.
            ...

    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.handler = None

    def __enter__(self) -> logging.Handler:
        if logging.root.level != logging.NOTSET:
            raise ValueError(
                "The root logger must have log level NOTSET to use the"
                " FileLogging context."
            )

        # Create the log handler for the file.
        handler = logging.FileHandler(self.file_path)

        # Configure the log handler.
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))

        # Attach the log handler to the root logger.
        logging.root.addHandler(handler)

        self.handler = handler

        return self.handler

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        logging.root.removeHandler(self.handler)
