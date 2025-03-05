import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import numpy as np
import pandas as pd
import joblib
from ensure import ensure_annotations
from box import ConfiqBox
from pathlib import Path
from typing import List, Dict, Any, Tuple
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfiqBox:
    """
    Read a yaml file and returns
    
    Args:
    path_to_yaml (str): path like input

    Raises:
          ValueError: If the file is empty
          e: empty file

    Returns:
          ConfiqBox: ConfiqBox type      
    """

    try: 
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfiqBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directory(path_to_directories: list, verbose=True):

    """
    Create list of directories

    Args:
            path_to_directories (list): list of path of directories
            ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"create Directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: Dict):
    """
    Save a dictionary to a json file

    Args:
        data (Dict): dictionary to save
        path (Path): path to save the json file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfiqBox:
    """
    Load a json file

    Args:
        path (Path): path to the json file

    Returns:
        Dict: dictionary
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfiqBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save a binary file

    Args:
        data (Any): data to save
        path (Path): path to save the binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file

    Args:
        path (Path): path to the binary file

    Returns:
        Any: data
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded successfully from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path (Path): path to the file

    Returns:
        int: size of the file
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
       
         return base64.b64encode(f.read())

