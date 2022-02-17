import torch.nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.roberta import RobertaModel

# assert isinstance(AutoModel, PreTrainedModel)

# ensure:
# for p in model.parameters():
#     p.requires_grad = True

# model = fn(model) or model, so returning model is optional


def freeze(model: PreTrainedModel):
    for p in model.parameters():
        p.requires_grad = False


def freeze_word_embeddings(model: PreTrainedModel):
    word_embeddings = model.get_input_embeddings()
    assert isinstance(word_embeddings, torch.nn.Module)
    for p in word_embeddings.parameters():
        p.requires_grad = False


def unfreeze_word_embeddings(model: PreTrainedModel):
    word_embeddings = model.get_input_embeddings()
    assert isinstance(word_embeddings, torch.nn.Module)
    for p in word_embeddings.parameters():
        p.requires_grad = True


def freeze_embeddings(model: PreTrainedModel):
    assert isinstance(model.base_model.embeddings, torch.nn.Module)
    for p in model.base_model.embeddings.parameters():
        p.requires_grad = False


def unfreeze_embeddings(model: PreTrainedModel):
    assert isinstance(model.base_model.embeddings, torch.nn.Module)
    for p in model.base_model.embeddings.parameters():
        p.requires_grad = True


def replace_word_embeddings(model: PreTrainedModel, wemb_model: PreTrainedModel):
    assert isinstance(wemb_model, PreTrainedModel)
    model.config.vocab_size = wemb_model.config.vocab_size
    model.set_input_embeddings(wemb_model.get_input_embeddings())
    model.tie_weights()


def replace_embeddings(model: PreTrainedModel, wemb_model: PreTrainedModel):
    assert isinstance(model.base_model.embeddings, torch.nn.Module)
    model.config.vocab_size = wemb_model.config.vocab_size
    model.base_model.embeddings = wemb_model.base_model.embeddings
    model.tie_weights()
