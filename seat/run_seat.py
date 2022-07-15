from argparse import Namespace
from csv import DictWriter
from logging import Logger
import numpy as np
import os
import re
from seat_utils import (
    clear_console,
    get_seat_logger,
    load_encodings,
    load_json,
    save_encodings,
    set_seed,
)
from seat_config import get_seat_args, TEST_EXT
from seat_encoders import load_model_and_tokenizer, get_encodings
import weat


def get_keys_to_sort_tests(test: str):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name and the strings in between them."""
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def check_availability(arg_str: str, allowed_set: list, item_type: str):
    """Given a comma-separated string of items, split on commas and check if all items are in `allowed_set`."""
    test_items = arg_str.split(",")
    for test_item in test_items:
        if test_item not in allowed_set:
            raise ValueError(f"Unknown {item_type}: {test_item}.")

    return test_items


def get_encoded_vectors(enc_path: str):
    encs = load_encodings(enc_path=enc_path)

    encs_targ1 = encs["targ1"]
    encs_targ2 = encs["targ2"]
    encs_attr1 = encs["attr1"]
    encs_attr2 = encs["attr2"]

    return encs_targ1, encs_targ2, encs_attr1, encs_attr2


def save_encoded_vectors(data: dict, encs_targ1: dict, encs_targ2: dict, encs_attr1: dict, encs_attr2: dict):
    """Save the encoded vectors in the dataset with another key name."""
    data["targ1"]["encs"] = encs_targ1
    data["targ2"]["encs"] = encs_targ2
    data["attr1"]["encs"] = encs_attr1
    data["attr2"]["encs"] = encs_attr2

    return data


def run_seat(args: Namespace, logger: Logger):
    """Parse args for seat to run and which models to evaluate.

    Args:
        args (Namespace): A parsed arguments.
        logger (Logger): A logger for checking process.
    """
    # set seed
    if args.seed >= 0:
        logger.info(f"Seed: {args.seed}")
        set_seed(args)

    # get all tests
    all_tests = sorted(
        [
            entry[: -len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith(".") and entry.endswith(TEST_EXT)
        ],
        key=get_keys_to_sort_tests,
    )
    logger.info(f"Found tests: {all_tests}")

    # check the available tests
    tests = (
        check_availability(arg_str=args.tests, allowed_set=all_tests, item_type="test")
        if args.tests is not None
        else all_tests
    )
    logger.info(f"Selected tests: {tests}")

    # check the available models
    available_models = (
        check_availability(arg_str=args.model_name, allowed_set=["bert", "roberta", "albert"], item_type="model")
        if args.model_name is not None
        else ["bert", "roberta", "albert"]
    )
    logger.info(f"Selected models: {available_models}")

    results = []
    for model_name in available_models:
        logger.info(f"Start to run the SEAT for {model_name}.")

        for test in tests:
            logger.info(f"Start to run {test} for {model_name}.")

            # get encoded file
            enc_path = os.path.join(args.exp_dir, f"{args.version}.h5" if args.version else f"{model_name}_{test}.h5")

            # load encoded vectors or test dataset to encode
            if not args.ignore_cached_encs and os.path.isfile(enc_path):
                logger.info(f"Load cached encodings: {enc_path}")
                encs_targ1, encs_targ2, encs_attr1, encs_attr2 = get_encoded_vectors(enc_path=enc_path)
            else:
                logger.info(f"Load data for SEAT: {os.path.join(args.data_dir, f'{test}{TEST_EXT}')}")
                test_data = load_json(os.path.join(args.data_dir, f"{test}{TEST_EXT}"))

                # load the model and tokenizer
                logger.info("Compute sentence encodings.")
                model, tokenizer = load_model_and_tokenizer(version=args.version, args=args)
                # get encodings
                encs_targ1, encs_targ2, encs_attr1, encs_attr2 = get_encodings(
                    data_keys=["targ1", "targ2", "attr1", "attr2"], data=test_data, model=model, tokenizer=tokenizer
                )

            # save encoded vectors in `test_data` with `data` key name
            encoded_data = save_encoded_vectors(
                data=test_data,
                encs_targ1=encs_targ1,
                encs_targ2=encs_targ2,
                encs_attr1=encs_attr1,
                encs_attr2=encs_attr2,
            )
            logger.info("Encoding is done.")
            if args.cache_encs:
                logger.info(f"Save the encodings to {enc_path}")
                save_encodings(encodings=encoded_data, enc_path=enc_path)

            # check the encoding and run the SEAT on the encodings
            enc = [e for e in dict.values(encoded_data["targ1"]["encs"])][0]
            rep_dim = enc.size if isinstance(enc, np.ndarray) else len(enc)
            logger.info("Start to run SEAT.")
            logger.info(f"A representation dimension: {rep_dim}")

            # get WEAT results and save them as a result dict
            effect_size, p_value = weat.run_test(
                encs=encoded_data, num_samples=args.num_samples, use_parametric=args.use_parametric, logger=logger
            )
            results.append(
                dict(
                    model=model_name,
                    version=args.version,
                    test=test,
                    p_value=p_value,
                    effect_size=effect_size,
                    num_targ1=len(encoded_data["targ1"]["encs"]),
                    num_targ2=len(encoded_data["targ2"]["encs"]),
                    num_attr1=len(encoded_data["attr1"]["encs"]),
                    num_attr2=len(encoded_data["attr2"]["encs"]),
                )
            )

        logger.info(f"Model: {model_name}")
        logger.info(f"Model version: {args.version}")
        for result in results:
            logger.info("\tTest {test}\tp-value: {p_value:.9f}\teffect-size: {effect_size:.2f}".format(**result))

    if args.results_path is not None:
        logger.info(f"Save the SEAT results to {args.results_path}")
        with open(file=args.results_path, mode="w") as res_fp:
            writer = DictWriter(f=res_fp, fieldnames=dict.keys(results[0]), delimiter="\t")
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    clear_console()

    args = get_seat_args()
    logger = get_seat_logger(args)
    run_seat(args=args, logger=logger)
