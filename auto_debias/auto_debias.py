from argparse import Namespace
from logging import Logger
import torch
from transformers.optimization import AdamW
from aa_config import get_aa_args
from aa_utils import (
    clear_console,
    get_aa_logger,
    prepare_model_and_tokenizer,
    load_words,
    clear_words,
    to_cuda,
    tokenize_prompts,
    JSDivergence,
    get_logits,
    overwrite_state_dict,
)
from torch.utils.data import DataLoader


def auto_debias(args: Namespace, logger: Logger):
    """Debias a pre-trained model using fine-tuning.

    Args:
        args (Namespace): A parsed arguments.
        logger (Logger): A logger for checking progress information.
    """
    # prepare pre-trained model and tokenizer
    logger.info(f"Prepare a pre-trained model and tokenizer: {args.model_version}")
    model, tokenizer = prepare_model_and_tokenizer(
        model_name=args.model_name, model_version=args.model_version, mode="auto-debias"
    )

    # get generated prompts
    logger.info(f"Load generated prompts: {args.data_dir + f'prompts_{args.model_version}_{args.debias_type}'}")
    prompts = load_words(path=args.data_dir + f"prompts_{args.model_version}_{args.debias_type}", mode="auto-debias")

    # get attribute words
    logger.info(f"Load attribute words for {args.debias_type}")
    if args.debias_type == "gender":
        _targ1_words = load_words(path=args.data_dir + "male.txt", mode="auto-debias")
        _targ2_words = load_words(path=args.data_dir + "female.txt", mode="auto-debias")
    elif args.debias_type == "race":
        _targ1_words = load_words(path=args.data_dir + "af_american.txt", mode="auto-debias")
        _targ2_words = load_words(path=args.data_dir + "eu_american.txt", mode="auto-debias")

    # clean target words
    logger.info(f"Remove words that contains an OOV tokens.")
    targ1_words, targ2_words = clear_words(
        _words1=_targ1_words, _words2=_targ2_words, tokenizer=tokenizer, mode="auto-debias"
    )

    # get prompt tokens
    logger.info("Get prompts for fine-tuning.")
    targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = tokenize_prompts(
        prompts=prompts, targ1_words=targ1_words, targ2_words=targ2_words, tokenizer=tokenizer
    )
    targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

    # whether or not to fine-tune a pre-defined specific vocab
    if args.finetune_vocab != "None":
        logger.info("Fine-tune a pre-defined specific vocab.")
        finetune_vocab = load_words(path=args.data_dir + args.finetune_vocab, mode="auto-debias")
        finetune_ids = tokenizer.convert_tokens_to_ids(finetune_vocab)

    # set dataloader containing batch index
    dataloader = DataLoader(
        dataset=[i for i in range(torch.Tensor.size(targ1_tokens.input_ids)[0])],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # set loss function
    logger.info(f"Set loss function and optimizer.")
    js_div_module = JSDivergence(reduction="batchmean")
    optimizer = AdamW(params=model.parameters(), lr=args.lr)

    for epoch in range(1, args.max_epochs + 1):
        # init loss for an epoch
        epoch_loss = 0.0
        # load batch data
        for iter, batch_idx in enumerate(dataloader):
            # init batch model inputs
            targ1_inputs = {}
            targ2_inputs = {}

            # get batch inputs with batch index
            for key in targ1_tokens.keys():
                targ1_inputs[key] = targ1_tokens[key][batch_idx]
                targ2_inputs[key] = targ2_tokens[key][batch_idx]
            targ1_batch_mask_idx = targ1_mask_idx[batch_idx]
            targ2_batch_mask_idx = targ2_mask_idx[batch_idx]

            # set gradients as zero
            optimizer.zero_grad()

            # get logits for [MASK] token
            if args.finetune_vocab != "None":
                targ1_logits = get_logits(
                    model=model,
                    inputs=targ1_inputs,
                    mask_idx=targ1_batch_mask_idx,
                    stereotype_ids=None,
                    mode="auto-debias",
                    finetune_ids=finetune_ids,
                )
                targ2_logits = get_logits(
                    model=model,
                    inputs=targ2_inputs,
                    mask_idx=targ2_batch_mask_idx,
                    stereotype_ids=None,
                    mode="auto-debias",
                    finetune_ids=finetune_ids,
                )
            else:
                targ1_logits = get_logits(
                    model=model,
                    inputs=targ1_inputs,
                    mask_idx=targ1_batch_mask_idx,
                    stereotype_ids=None,
                    mode="auto-debias",
                    finetune_ids=None,
                )
                targ2_logits = get_logits(
                    model=model,
                    inputs=targ2_inputs,
                    mask_idx=targ2_batch_mask_idx,
                    stereotype_ids=None,
                    mode="auto-debias",
                    finetune_ids=None,
                )

            # get JS-Divergence value for [MASK] token as loss
            batch_loss = js_div_module.forward(net1_logits=targ1_logits, net2_logits=targ2_logits)

            # compute gradients of current loss for a batch
            batch_loss.backward()
            # update model params with a single step
            optimizer.step()
            # set gradients as zero
            optimizer.zero_grad()
            # record loss for a batch every 100 iterations
            if (iter + 1) % 100 == 0:
                logger.info(f"Epoch: {epoch}, Iter: {iter + 1}/{len(dataloader)}, Batch Loss: {batch_loss:.4f}")
            # accumulate batch loss
            epoch_loss += batch_loss

        # record loss for a batch at the last iteration
        logger.info(f"Epoch: {epoch}, Iter: {iter + 1}/{len(dataloader)}, Batch Loss: {batch_loss:.4f}")

        # after an epoch
        logger.info(f"Epoch: {epoch} - Epoch Loss: {epoch_loss / len(dataloader):.4f}")
        if epoch == args.max_epochs:
            # save debiased model and tokenizer for an epoch
            logger.info(
                f"Save debiased model: {args.output_dir + f'{args.model_name}_{args.run_name}_{args.debias_type}_epoch:{epoch}'}"
            )
            # overwrite weights from MaskedLM to SequenceClassification
            new_model = overwrite_state_dict(trained_model=model, args=args)
            # save SequenceClassification model
            new_model.save_pretrained(
                args.output_dir + f"{args.model_name}_{args.run_name}_{args.debias_type}_epoch:{epoch}"
            )
            tokenizer.save_pretrained(
                args.output_dir + f"{args.model_name}_{args.run_name}_{args.debias_type}_epoch:{epoch}"
            )


if __name__ == "__main__":
    clear_console()

    args = get_aa_args()
    logger = get_aa_logger(args)
    auto_debias(args=args, logger=logger)
