import os
import logging
from logging import Logger
from transformers.training_args import TrainingArguments


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_glue_logger(training_args: TrainingArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(
        fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(training_args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(filename=training_args.output_dir + f"glue_{training_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run number: {training_args.run_name}")

    return logger
