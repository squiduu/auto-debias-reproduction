import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import random
from argparse import Namespace
from typing import Dict, List, Tuple, Union
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from logging import Logger
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.models.albert.modeling_albert import AlbertForMaskedLM, AlbertForSequenceClassification

MALE_WORDS = [
    "fathers",
    "actor",
    "prince",
    "men",
    "gentlemen",
    "sir",
    "brother",
    "his",
    "king",
    "husband",
    "dad",
    "males",
    "sir",
    "him",
    "boyfriend",
    "he",
    "hero",
    "kings",
    "brothers",
    "son",
    "sons",
    "himself",
    "gentleman",
    "his",
    "father",
    "male",
    "man",
    "grandpa",
    "boy",
    "grandfather",
]
FEMALE_WORDS = [
    "mothers",
    "actress",
    "princess",
    "women",
    "ladies",
    "madam",
    "sister",
    "her",
    "queen",
    "wife",
    "mom",
    "females",
    "miss",
    "her",
    "girlfriend",
    "she",
    "heroine",
    "queens",
    "sisters",
    "daughter",
    "daughters",
    "herself",
    "lady",
    "hers",
    "mother",
    "female",
    "woman",
    "grandma",
    "girl",
    "grandmother",
]
# african american
AFA = [
    "black",
    "african",
    "black",
    "africa",
    "africa",
    "africa",
    "black people",
    "african people",
    "black people",
    "the africa",
]
# european american
EUA = [
    "caucasian",
    "caucasian",
    "white",
    "america",
    "america",
    "europe",
    "caucasian people",
    "caucasian people",
    "white people",
    "the america",
]


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_aa_logger(args: Namespace) -> Logger:
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
    os.makedirs(args.output_dir, exist_ok=True)
    if "generate" in sys.argv[0]:
        file_hdlr = logging.FileHandler(filename=args.output_dir + f"prompt_{args.run_name}.log")
    elif "auto" in sys.argv[0]:
        file_hdlr = logging.FileHandler(filename=args.output_dir + f"auto_{args.run_name}.log")
    elif "aa" in sys.argv[0]:
        file_hdlr = logging.FileHandler(filename=args.output_dir + f"aa_{args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {args.run_name}")
    # record arguments and hparams
    logger.info(f"Config: {vars(args)}")

    return logger


def prepare_model_and_tokenizer(
    model_name: str, model_version: str, mode: str
) -> Union[
    Tuple[
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    ],
    Tuple[
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    ],
]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name (str): A name of pre-trained model.
        model_version (str): A version of pre-trained model.
        mode (str): A status of either generating prompts or debiasing models.
    """
    if model_name == "bert":
        model_class = BertForMaskedLM
        tokenizer_class = BertTokenizer
    elif model_name == "roberta":
        model_class = RobertaForMaskedLM
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertForMaskedLM
        tokenizer_class = AlbertTokenizer

    # get tokenizer because it is common for mode
    tokenizer = tokenizer_class.from_pretrained(model_version)

    if mode == "auto-debias":
        model = model_class.from_pretrained(model_version)

        model.train().cuda()

    elif mode == "aa-debias":
        freezing_model = model_class.from_pretrained(model_version)
        tuning_model = model_class.from_pretrained(model_version)

        freezing_model.eval().cuda()
        tuning_model.train().cuda()

    elif mode == "prompt":
        model = model_class.from_pretrained(model_version)

        model = DataParallel(model)
        model.eval().cuda()

    # move all model parameters to the GPU
    if mode in ["auto-debias", "generate"]:
        return model, tokenizer

    elif mode == "aa-debias":
        return freezing_model, tuning_model, tokenizer


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()

        self.reduction = reduction

    def forward(self, net1_logits: torch.FloatTensor, net2_logits: torch.FloatTensor) -> torch.FloatTensor:
        net1_dist = F.softmax(input=net1_logits, dim=1)
        net2_dist = F.softmax(input=net2_logits, dim=1)

        avg_dist = (net1_dist + net2_dist) / 2.0

        js_div_value = 0.0
        js_div_value += F.kl_div(input=F.log_softmax(net1_logits, dim=1), target=avg_dist, reduction=self.reduction)
        js_div_value += F.kl_div(input=F.log_softmax(net2_logits, dim=1), target=avg_dist, reduction=self.reduction)

        return js_div_value / 2.0


def load_words(path: str, mode: str) -> List[str]:
    if mode == "prompt":
        list = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                list.append(lines[i].strip().split(sep=" ")[0])

    elif mode in ["stereotype", "auto-debias", "aa-debias"]:
        list = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                list.append(lines[i].strip())

    return list


def clear_words(
    _words1: List[str],
    _words2: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    mode: str,
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """Remove the input word if the word contains the out-of-vocabulary token.

    Args:
        _words1 (List[str]): Input words to check the out-of-vocabulary.
        _words2 (List[str]): Input words to check the out-of-vocabulary.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
        mode (str): A status of either generating prompts or debiasing models.

    Returns:
        Union[List[str], Tuple[List[str], List[str]]]: _description_
    """
    if mode in ["prompt", "stereotype"] and _words2 is None:
        words = []
        for i in range(len(_words1)):
            if tokenizer.convert_tokens_to_ids(_words1[i]) != tokenizer.unk_token_id:
                words.append(_words1[i])

        return words

    elif mode in ["auto-debias", "aa-debias"] and _words2 is not None:
        words1 = []
        words2 = []
        for i in range(len(_words1)):
            if (
                tokenizer.convert_tokens_to_ids(_words1) != tokenizer.unk_token_id
                and tokenizer.convert_tokens_to_ids(_words2) != tokenizer.unk_token_id
            ):
                words1.append(_words1[i])
                words2.append(_words2[i])

        return words1, words2


def tokenize_ith_prompts(
    prompts: List[str],
    targ1_word: str,
    targ2_word: str,
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create prompts with i-th target concept word and tokenize them.

    Args:
        prompts (List[str]): A total prompt words.
        targ1_word (str): An i-th target 1 word.
        targ2_word (str): An i-th target 2 word.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 sentences
    for i in range(len(prompts)):
        targ1_sents.append(targ1_word + " " + prompts[i] + " " + tokenizer.mask_token)
        targ2_sents.append(targ2_word + " " + prompts[i] + " " + tokenizer.mask_token)

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")
    # del targ1 and targ2 sentences
    del targ1_sents, targ2_sents

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def tokenize_prompts(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create prompts with target concept word and tokenize them.

    Args:
        prompts (List[str]): A total prompt words.
        targ1_words (List[str]): An i-th target 1 word.
        targ2_words (List[str]): An i-th target 2 word.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 sentences
    for i in range(len(prompts)):
        for j in range(len(targ1_words)):
            targ1_sents.append(targ1_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")
            targ2_sents.append(targ2_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")

    # del targ1 and targ2 sentences
    del targ1_sents, targ2_sents

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def to_cuda(targ1_tokens: BatchEncoding, targ2_tokens: BatchEncoding) -> Tuple[BatchEncoding, BatchEncoding]:
    for key in targ1_tokens.keys():
        targ1_tokens[key] = torch.Tensor.cuda(targ1_tokens[key])
        targ2_tokens[key] = torch.Tensor.cuda(targ2_tokens[key])

    return targ1_tokens, targ2_tokens


def get_batch_inputs(
    batch_idx: int,
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    args: Namespace,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor], np.ndarray, np.ndarray]:
    """Slice all inputs as `batch_size`.

    Args:
        idx (int): An index for batch.
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        args (Namespace): A parsed arguments.

    Returns:
        targ1_inputs (Dict[str, torch.LongTensor]): Tokens for target 1 concepts sliced as `batch_size`.
        targ2_inputs (Dict[str, torch.LongTensor]): Tokens for target 2 concepts sliced as `batch_size`.
        targ1_local_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
        targ2_local_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
    """
    targ1_inputs = {}
    targ2_inputs = {}

    try:
        for key in targ1_tokens.keys():
            # slice to batch size
            targ1_inputs[key] = targ1_tokens[key][args.batch_size * batch_idx : args.batch_size * (batch_idx + 1)]
            targ2_inputs[key] = targ2_tokens[key][args.batch_size * batch_idx : args.batch_size * (batch_idx + 1)]

        targ1_local_mask_idx = targ1_mask_idx[args.batch_size * batch_idx : args.batch_size * (batch_idx + 1)]
        targ2_local_mask_idx = targ2_mask_idx[args.batch_size * batch_idx : args.batch_size * (batch_idx + 1)]

    except IndexError:
        for key in targ1_tokens.keys():
            # get rest of batches
            targ1_inputs[key] = targ1_tokens[key][args.batch_size * (batch_idx + 1) :]
            targ2_inputs[key] = targ2_tokens[key][args.batch_size * (batch_idx + 1) :]

        targ1_local_mask_idx = targ1_mask_idx[args.batch_size * (batch_idx + 1) :]
        targ2_local_mask_idx = targ2_mask_idx[args.batch_size * (batch_idx + 1) :]

    return targ1_inputs, targ2_inputs, targ1_local_mask_idx, targ2_local_mask_idx


def get_logits(
    freezing_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tuning_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    inputs: Dict[str, torch.LongTensor],
    mask_idx: np.ndarray,
    stereotype_ids: List[int],
    mode: str,
    finetune_ids: List[int] = None,
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
    """Get logits corresponding to stereotype words at [MASK] token position.

    Args:
        freezing_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for freezing.
        tuning_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for fine-tuning.
        inputs (Dict[str, torch.LongTensor]): Tokenized prompt inputs with a [MASK] token.
        mask_idx (np.ndarray): An index of a [MASK] token in tokenized prompt inputs.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        mode (str): A status of either generating prompts or debiasing models.
        finetune_ids (List[int], optional): Whether or not to fine-tune specific vocab. Default to None.
    """
    if mode == "auto-debias" and freezing_model is None and tuning_model is not None:
        outputs = tuning_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        if finetune_ids is not None:
            _logits = outputs.logits[torch.arange(torch.Tensor.size(outputs.logits)[0]), mask_idx]
            logits = _logits[:, finetune_ids]
        else:
            logits = outputs.logits[torch.arange(torch.Tensor.size(outputs.logits)[0]), mask_idx]

        return logits

    elif mode == "aa-debias" and freezing_model is not None and tuning_model is not None:
        freezing_outputs = freezing_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        tuning_outputs = tuning_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        freezing_logits = freezing_outputs.logits[torch.arange(torch.Tensor.size(freezing_outputs.logits)[0]), mask_idx]
        tuning_logits = tuning_outputs.logits[torch.arange(torch.Tensor.size(tuning_outputs.logits)[0]), mask_idx]

        return freezing_logits, tuning_logits

    elif mode == "generate" and freezing_model is not None and tuning_model is None:
        outputs = freezing_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        # extract logits only for stereotype words
        logits = outputs.logits[np.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx][:, stereotype_ids]

        return logits


def get_cosine_similarity(logits1: torch.FloatTensor, logits2: torch.FloatTensor) -> torch.FloatTensor:
    cos_sim = F.cosine_similarity(logits1, logits2)

    return cos_sim.mean()


def get_js_div_values(
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    stereotype_ids: List[int],
    js_div_module: JSDivergence,
    args: Namespace,
) -> List[np.ndarray]:
    """Calculate JS-Divergence values and accumulate them for all prompts of i-th target concept word.

    Args:
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        js_div_module (JSDivergence): A JS-Divergence module.
        args (Namespace): A parsed arguments.

    Returns:
        js_div_values (List[np.ndarray]): _description_
    """
    js_div_values = []
    # send all tokens to cuda for dataparallel
    targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

    for batch_idx in range(torch.Tensor.size(targ1_tokens["input_ids"])[0] // args.batch_size + 1):
        # slice inputs as batch size
        targ1_inputs, targ2_inputs, targ1_batch_mask_idx, targ2_batch_mask_idx = get_batch_inputs(
            batch_idx=batch_idx,
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            args=args,
        )

        # get logits of stereotype words
        targ1_logits = get_logits(
            model=model,
            inputs=targ1_inputs,
            mask_idx=targ1_batch_mask_idx,
            stereotype_ids=stereotype_ids,
            mode="generate",
            finetune_ids=None,
        )
        targ2_logits = get_logits(
            model=model,
            inputs=targ2_inputs,
            mask_idx=targ2_batch_mask_idx,
            stereotype_ids=stereotype_ids,
            mode="generate",
            finetune_ids=None,
        )

        # get JS-Divergence value for two networks
        js_div_value = js_div_module.forward(net1_logits=targ1_logits, net2_logits=targ2_logits)
        js_div_sum = np.sum(js_div_value.detach().cpu().numpy(), axis=1)
        # accumulate all JS-Divergence values
        js_div_values += list(js_div_sum)

        del targ1_logits, targ2_logits, js_div_value

    return js_div_values


def get_prompt_js_div(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    stereotype_ids: List[int],
    js_div_module: JSDivergence,
    args: Namespace,
) -> np.ndarray:
    """Get JS-Divergence values for all prompts of all target concept words about bias.

    Args:
        prompts (List[str]): Candidate words for prompts.
        targ1_words (List[str]): Words for target 1 concepts.
        targ2_words (List[str]): Words for target 2 concepts.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        js_div_module (JSDivergence): A JS-Divergence module.
        args (Namespace): A parsed arguments.

    Returns:
        accum_prompt_js_div_values (np.ndarray): Accumulated JS-Divergence values for all prompts.
    """
    prompt_js_div_values = []
    # for i in tqdm(iterable=range(len(targ1_words))):
    for i in tqdm(range(len(targ1_words))):
        # create all possible prompts combination for i-th target concept word
        targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = tokenize_ith_prompts(
            prompts=prompts,
            targ1_word=targ1_words[i],
            targ2_word=targ2_words[i],
            tokenizer=tokenizer,
        )
        # get JS-Divergence values of i-th target concept word
        js_div_values = get_js_div_values(
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            model=model,
            stereotype_ids=stereotype_ids,
            js_div_module=js_div_module,
            args=args,
        )
        # accumulate all target concept words
        prompt_js_div_values.append(js_div_values)
    prompt_js_div_values = np.array(prompt_js_div_values)
    accum_prompt_js_div_values = np.mean(prompt_js_div_values, axis=0)

    return accum_prompt_js_div_values


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
    if args.deterministic == True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def overwrite_state_dict(
    trained_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM], args: Namespace
) -> Union[BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification]:
    """Extract and transfer only the trained weights of the layer matching the new model.

    Args:
        trained_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A debiased model.
        args (Namespace): A parsed arguments.
    """
    if type(trained_model) == BertForMaskedLM:
        model_class = BertForSequenceClassification
    elif type(trained_model) == RobertaForMaskedLM:
        model_class = RobertaForSequenceClassification
    else:
        model_class = AlbertForSequenceClassification

    # get initialized pre-trained model
    new_model = model_class.from_pretrained(args.model_version)

    # get initialized pre-trained model weights
    new_model_dict = new_model.state_dict()
    # filter out unnecessary keys in debiased masked model
    trained_model_dict = {k: v for k, v in trained_model.state_dict().items() if k in new_model_dict}
    # overwrite entries in the existing initialized state dict
    new_model_dict.update(trained_model_dict)

    # overwrite updated weights
    new_model.load_state_dict(new_model_dict)

    return new_model
