import h5py
import logging
import json
import os
import random
import numpy as np
import torch
from logging import Logger
from argparse import Namespace

CATEGORY = "category"


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"

    os.system(command)


def get_seat_logger(args: Namespace) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(args.log_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=args.log_dir + f"seat_{args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run number: {args.run_name}")
    # record arguments and hparams
    logger.info(f"Config: {vars(args)}")

    return logger


def load_encodings(enc_path: str):
    """Load cached encoding vectors from a model.

    Args:
        enc_path (str): A cached encoding vectors.
    """
    encs = dict()
    with h5py.File(name=enc_path, mode="r") as enc_fp:
        for split_name, split in enc_fp.items():
            split_d, split_exs = {}, {}
            for ex, enc in h5py.File.items(split):
                if ex == CATEGORY:
                    split_d[ex] = enc[()]
                else:
                    split_exs[ex] = enc[:]
            split_d["encs"] = split_exs
            encs[split_name] = split_d

    return encs


def load_json(json_path: str):
    """Load from .json file. We expect a certain format later, so do some post processing."""
    all_data = json.load(open(file=json_path, mode="r"))
    # data = {}
    # for key, value in dict.items(all_data):
    #     examples = value["examples"]
    #     data[key] = examples
    #     value["examples"] = examples

    return all_data


def save_encodings(encodings: dict, enc_path: str):
    """Save encodings to file."""
    with h5py.File(name=enc_path, mode="w") as enc_fp:
        for split_name, split_d in encodings.items():
            split = enc_fp.create_group(split_name)
            split[CATEGORY] = split_d["category"]
            for ex, enc in split_d["encs"].items():
                split[ex] = enc


def set_seed(args: Namespace):
    """Set a seed for complete reproducibility.

    Args:
        args (Namespace): A parsed arguments.
    """
    # for python
    random.seed(args.seed)
    # for numpy
    np.random.seed(args.seed)
    # for torch
    torch.manual_seed(args.seed)
    # for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # set deterministic as False if the runtime is too long
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
