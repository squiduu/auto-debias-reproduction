from typing import Dict, List, Tuple, Union
from numpy import ndarray
import torch
from argparse import Namespace
from transformers import AlbertPreTrainedModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.modeling_albert import AlbertForMaskedLM


def load_model_and_tokenizer(
    version: str, args: Namespace
) -> Tuple[
    Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
]:
    """Load a pre-trained or checkpoint model as evaluation and corresponding tokenizer.

    Args:
        version (str): A version of pre-trained model.
        args (Namespace): A parsed arguments.

    Returns:
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained or checkpoint model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    if args.use_ckpt:
        # load from checkpoint
        if "bert" in version:
            model = BertForMaskedLM.from_pretrained(args.ckpt_dir)
        elif "roberta" in version:
            model = RobertaForMaskedLM.from_pretrained(args.ckpt_dir)
        else:
            model = AlbertForMaskedLM.from_pretrained(args.ckpt_dir)
    else:
        # load from huggingface
        if "bert" in version:
            model = BertForMaskedLM.from_pretrained(version)
        elif "roberta" in version:
            model = RobertaForMaskedLM.from_pretrained(version)
        else:
            model = AlbertPreTrainedModel.from_pretrained(version)

    # load only pre-trained tokenizer because we did not change the special tokens
    if "bert" in version:
        tokenizer = BertTokenizer.from_pretrained(version)
    elif "roberta" in version:
        tokenizer = RobertaTokenizer.from_pretrained(version)
    else:
        tokenizer = AlbertTokenizer.from_pretrained(version)

    model.eval()

    return model, tokenizer


def get_encodings(
    data_keys: List[str],
    data: Dict[str, str],
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[Dict[str, ndarray], Dict[str, ndarray], Dict[str, ndarray], Dict[str, ndarray]]:
    """Encode the input data using the PreTrainedTokenizer and PreTrainedModel.

    Args:
        data_keys (List[str]): Key names for iteration.
        data (Dict[str, str]): An input data.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A Pre-trained PreTrainedModel model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A Pre-trained PreTrainedTokenizer.

    Returns:
        encs_targ1, encs_targ2, encs_attr1, encs_attr2: Encodings corresponding to the encoderd and tokenizer.
    """
    for i, key in enumerate(data_keys):
        encs = {}
        for sent in data[key]["examples"]:
            tokens = tokenizer.tokenize(sent)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(tokens)

            # make model input tensors
            tokens_tensor = torch.tensor([token_ids])
            segments_tensor = torch.tensor([segment_ids])

            # get BERT outputs
            if type(model) == BertForMaskedLM:
                outputs = model.bert.forward(input_ids=tokens_tensor, token_type_ids=segments_tensor)
            elif type(model) == RobertaForMaskedLM:
                outputs = model.roberta.forward(input_ids=tokens_tensor, token_type_ids=segments_tensor)
            else:
                outputs = model.albert.forward(input_ids=tokens_tensor, token_type_ids=segments_tensor)
            # extract the last rep of the [CLS] token
            encs[sent] = torch.detach(outputs.last_hidden_state[:, 0, :]).reshape(-1).numpy()

        if i == 0:
            encs_targ1 = encs
        elif i == 1:
            encs_targ2 = encs
        elif i == 2:
            encs_attr1 = encs
        else:
            encs_attr2 = encs

    return encs_targ1, encs_targ2, encs_attr1, encs_attr2
