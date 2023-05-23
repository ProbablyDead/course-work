from transformers import GPT2LMHeadModel, GPT2Tokenizer
from AI import Chat
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

PT_FILE_PATH = "model_state.pt"
SAVED_MODEL = "sources/model/pretrained"
PRETRAINED_MODEL = "gpt2"
DATASET_PATH = "../sources/datasets/discussions_debatepedia.json"
DEFAULT_MESSAGE = "Is/was the passage of a $700b bill urgent?"
EPOCHS = 10
UNITS_TO_LEARN = 10000
BATCH_SIZE = 32


def train(chat_data, model, tokenizer, optim, device) -> None:
    """
    Trains the GPT2 language model.

    Args:
        chat_data (DataLoader): DataLoader object containing the chat data.
        model (GPT2LMHeadModel): Pretrained GPT2 language model.
        tokenizer (GPT2Tokenizer): Tokenizer object for GPT2.
        optim (Adam): Optimizer object.
        device (str): Device to use for training.

    Returns:
        None
    """
    epochs = EPOCHS

    for _ in tqdm.tqdm(range(epochs)):
        for unit, a in chat_data:
            unit = unit.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(unit, attention_mask=a, labels=unit).loss
            loss.backward()
            optim.step()

        model.save_pretrained(SAVED_MODEL)
        tokenizer.save_pretrained(SAVED_MODEL)


def prepare_default_model() -> None:
    """
    Prepares the default GPT2 language model for training.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.add_special_tokens({"pad_token": "<pad>",
                                  "bos_token": "<startofstring>",
                                  "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>:"])

    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    chat_data = Chat.Chat(DATASET_PATH, tokenizer)
    chat_data = DataLoader(chat_data, batch_size=BATCH_SIZE)

    model.train()

    optim = Adam(model.parameters(), lr=1e-3)

    train(chat_data, model, tokenizer, optim, device)
