"""Utilities for writing tests."""

import pkg_resources
import numpy as np
import torch


def copy_pkg_resource_to_disk(pkg: str, src: str, dst: str) -> None:
    """Copy a package resource to disk.

    Copy the resource from the package ``pkg`` at the path ``src`` to
    disk at the path ``dst``.

    Parameters
    ----------
    pkg : str
        The package holding the resource.
    src : str
        The source path for the resource in ``pkg``.
    dst : str
        The destination path for the resource on disk.

    Notes
    -----
    This function is primarily useful for testing code that requires
    resources to be written on disk, when those test fixtures are
    shipped in the package.
    """
    with pkg_resources.resource_stream(pkg, src) as src_file, open(
        dst, "wb"
    ) as dst_file:
        dst_file.write(src_file.read())


def row_in_df(row, df):
    """A helper function for checking that a row is in a pandas DataFrame"""
    for index, (df_row) in df.iterrows():
        if np.array_equal(row.values, df_row.values):
            return True
    return False


# some utilities with dealing with tensors in dictionaries for counting purposes
def torch_tensor_in_dict_keys(tensor: torch.tensor, index_dict: dict) -> bool:
    for key in list(index_dict.keys()):
        if torch.all(key.eq(tensor)):
            return True
    return False


def add_count_to_torch_tensor_in_dict_keys(
    tensor_to_match: torch.tensor, index_dict: dict
) -> bool:
    for key in list(index_dict.keys()):
        if torch.all(key.eq(tensor_to_match)):
            index_dict[key] += 1
            return index_dict
