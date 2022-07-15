import argparse

TEST_EXT = ".jsonl"


def get_seat_args():
    """Helper function for handling argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run specified SEAT on the specified models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tests",
        type=str,
        help=f"WEAT tests to run (a comma-separated list; test files should be in `data_dir` and have corresponding \
            names, with extension {TEST_EXT}). Default: all tests.",
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--seed", type=int, help="Random seed", default=1111)
    parser.add_argument("--log_dir", type=str, help="A directory to log to.")
    parser.add_argument("--results_path", type=str, help="Path where the .tsv results files will be written")
    parser.add_argument(
        "--ignore_cached_encs",
        action="store_true",
        help="If set, ignore existing encodings and encode from scratch.",
    )
    parser.add_argument("--cache_encs", action="store_true", help="If set, do not cache encodings to disk.")
    parser.add_argument("--data_dir", type=str, help="Directory containing examples for each test", default="tests")
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="Directory from which to load and save vectors. Files should be stored as h5py files.",
        default="out",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of permutation test samples used when estimate p-values (exact test is used if there are \
            fewer than this many permutations)",
        default=100000,
    )
    parser.add_argument(
        "--use_parametric", action="store_true", help="Use parametric test (normal assumption) to compute p-values."
    )
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU to encode sentences.")
    parser.add_argument("--run_name", type=str, help="A run number for recording.")
    parser.add_argument("--use_ckpt", action="store_true", help="Whether or not to use checkpoint.")
    parser.add_argument("--ckpt_dir", type=str, help="A directory of checkpoint containing config.json file.")
    parser.add_argument("--version", type=str, help="A model version from HuggingFace.")
    parser.add_argument("--deterministic", action="store_true", help="Whether or not to use checkpoint.")

    return parser.parse_args()
