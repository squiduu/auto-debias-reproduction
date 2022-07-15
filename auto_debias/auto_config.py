import argparse


def get_aa_args():
    parser = argparse.ArgumentParser()

    # for common
    parser.add_argument("--model_version", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--model_name", default="bert", type=str, choices=["bert", "roberta", "gpt2"]
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--debias_type", default="gender", type=str, choices=["gender", "race"]
    )
    parser.add_argument("--run_name", default="run00", type=str)
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument(
        "--mode",
        default="prompt",
        type=str,
        choices=["prompt", "auto-debias", "aa-debias"],
    )
    parser.add_argument("--deterministic", default=False, type=bool)

    # for prompting
    parser.add_argument("--max_prompt_len", default=5, type=int)
    parser.add_argument("--top_k", default=100, type=int)

    # for dataloader
    parser.add_argument("--num_workers", default=4, type=int)

    # for modeling
    parser.add_argument("--lr", default=2e-5, type=float)

    # for trainer
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--output_dir", default="./out/", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--finetune_vocab", default=None, type=str)

    return parser.parse_args()
