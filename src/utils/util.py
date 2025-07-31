"""
Utilities
"""

import os
from typing import Callable
from glob import glob

from torch.utils.tensorboard.writer import SummaryWriter
from common.constant import VIEW_MAP

def calculate_accuracy(true_labels, predicted_labels) -> float:
    """
    Calculate the accuracy of a set of prediction labels

    Args:
        true_labels (Iterable[str]): The actual labels
        predicted_labels (Iterable[str]): The predicted labels
    
    Returns:
        The accuracy as a percentage in range [0.0, 1.0]
    """
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy



def assert_exists(path:str, err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that there is a file or directory at the path
    
    Args:
        path (str): The path to check
        err_type (str -> Exception optional default: OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not exist
    """
    if not os.path.exists(os.path.abspath(path)):
        raise err_type(f'File/Folder "{os.path.abspath(path)}" does not exists')

def assert_is_dir(path:str,
                  err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that the path points to a directory
    
    Args:
        path (str): The path to check
        err_type (str -> Exception optional default: OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not point to a directory
    """
    if not os.path.exists(path):
        raise err_type(f'File/Folder "{path}" does not exists')

def assert_is_file(path:str,
                   readable:bool=False,
                   writable:bool=False,
                   err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that the path points to a file
    
    Args:
        path (str): The path to check
        readable (bool optional default: False): If true the file
            must additionally have read permission
        writable (bool optional default: False): If true the file
            must additionally have write permission
        err_type (str -> Exception optional default:OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not point to a file or the file
        it points to does not have the correct permissions
    """
    if not os.path.isfile(path):
        raise err_type(f'"{path}" is not a file')
    if readable and not os.access(path, os.R_OK):
        raise err_type(f'"{path}" does not have read permission')
    if writable and not os.access(path, os.W_OK):
        raise err_type(f'"{path}" does not have write permission')

def return_last_file(path:str):
    """
    Go through folders and return index of last folder
    
    Args:
        path (str):root directory to check
    Returns:
        index (int)
    """
    paths = glob(path)
    return len(paths)+1


def log(*args, **kwargs):
    # if params.verbose:
    print(*args, **kwargs)

def modify_view_map(info_train, info_val, info_test, combine_list ):
    """
    Modifies list of view labels for train, val, and test dfs by combining views specified in seperated dictionaries listed in combine_list
    
    Args:
        info_train (DataFrame): Data Info of Train
        info_val (DataFrame): Data Info of Val
        info_test (DataFrame): Data Info of Test
        combine_list (list[dict]): list of dictionaries, each ditionary's key is the combined view name and value is the detailed view names to be combined
    Returns:
        info_train (DataFrame): Updated View Mapping of Train Data
        info_val (DataFrame): Updated View Mapping of Val Data
        info_test (DataFrame): Updated View Mapping Test Data
    """

    for combine_dictionary in combine_list:
        original_view = list(combine_dictionary.values())[0]
        combine_view = list(combine_dictionary.keys())[0]
        info_train.loc[info_train['label'].isin(original_view), 'label'] = combine_view
        info_val.loc[info_val['label'].isin(original_view), 'label'] = combine_view
        info_test.loc[info_test['label'].isin(original_view), 'label'] = combine_view
        
    return info_train, info_val, info_test

        
