import numpy as np
from logging import Logger
from argparse import Namespace
from aa_config import get_aa_args
from aa_utils import (
    clear_console,
    get_aa_logger,
    prepare_model_and_tokenizer,
    JSDivergence,
    load_words,
    clear_words,
    get_prompt_js_div,
    MALE_WORDS,
    FEMALE_WORDS,
    AFA,
    EUA,
)


def generate_prompts(args: Namespace, logger: Logger):
    """Generate prompts and save them using JS-Divergence.

    Args:
        args (Namespace): A parsed arguments.
        logger (Logger): A logger for checking progress information.
    """
    # prepare pre-trained model and tokenizer
    logger.info(f"Prepare pre-trained model and tokenizer: {args.model_version}")
    model, tokenizer = prepare_model_and_tokenizer(
        model_name=args.model_name, model_version=args.model_version, mode=args.mode
    )

    # load and tokenize stereotype words
    logger.info(
        f"Load and tokenize stereotype words: {args.data_dir + 'stereotype_words.txt'}"
    )
    stereotype_words = clear_words(
        _words1=load_words(
            path=args.data_dir + "stereotype_words.txt", mode="stereotype"
        ),
        _words2=None,
        tokenizer=tokenizer,
        mode="stereotype",
    )
    stereotype_ids = tokenizer.convert_tokens_to_ids(stereotype_words)

    # load prompt words
    logger.info(
        f"Load and tokenize prompt words: {args.data_dir + 'wiki_words_5000.txt'}"
    )
    prompt_words = clear_words(
        _words1=load_words(path=args.data_dir + "wiki_words_5000.txt", mode=args.mode),
        _words2=None,
        tokenizer=tokenizer,
        mode=args.mode,
    )
    # init prompts
    current_prompts = prompt_words

    # create and open prompt file
    logger.info(
        f"Create and open prompt file: {args.data_dir + f'prompts_{args.model_version}_{args.debias_type}'}"
    )
    prompt_file = open(
        file=args.data_dir + f"prompts_{args.model_version}_{args.debias_type}",
        mode="w",
    )

    # init js-divergence
    js_div_module = JSDivergence(reduction="none")

    # calculate js-divergence for prompts
    logger.info(
        f"Get JS-Divergence values for all prompts about {args.debias_type} bias."
    )
    for i in range(args.max_prompt_len):
        logger.info(f"Maximum prompt length: {i + 1}")
        logger.info(f"No. of prompts: {len(current_prompts)}")

        if args.debias_type == "gender":
            current_prompts_js_div_values = get_prompt_js_div(
                prompts=current_prompts,
                targ1_words=MALE_WORDS,
                targ2_words=FEMALE_WORDS,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=stereotype_ids,
                js_div_module=js_div_module,
                args=args,
            )
        elif args.debias_type == "race":
            current_prompts_js_div_values = get_prompt_js_div(
                prompts=current_prompts,
                targ1_words=AFA,
                targ2_words=EUA,
                model=model,
                tokenizer=tokenizer,
                stereotype_ids=stereotype_ids,
                js_div_module=js_div_module,
                args=args,
            )

        logger.info(f"Select {args.top_k} prompts.")
        selected_prompts = np.array(current_prompts)[
            np.argsort(current_prompts_js_div_values)[::-1][: args.top_k]
        ]

        logger.info(f"Write {args.top_k} prompts to the file.")
        for selected_prompt in selected_prompts:
            prompt_file.write(selected_prompt)
            prompt_file.write("\n")

        logger.info("Create temporary prompts.")
        temp_prompts = []
        for selected_prompt in selected_prompts:
            for prompt_word in prompt_words:
                temp_prompts.append(selected_prompt + " " + prompt_word)

        logger.info("Update current prompts.")
        current_prompts = temp_prompts

    logger.info(f"Save and close prompt file.\n")
    prompt_file.close()


if __name__ == "__main__":
    clear_console()

    args = get_aa_args()
    logger = get_aa_logger(args)
    generate_prompts(args=args, logger=logger)
