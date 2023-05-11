from transformers import GPT2LMHeadModel, GPT2Tokenizer
from AI import Chat
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

PT_FILE_PATH = "model_state.pt"
PRETRAINED_MODEL = "gpt2"
DATASET_PATH = "sources/datasets/humor_funny.json"
DEFAULT_MESSAGE = "joe biden"
EPOCHS = 10
UNITS_TO_LEARN = 10000
MAX_REQUEST_LENGTH = 40
BATCH_SIZE = 32


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(chatData, model, optim, device):
    epochs = EPOCHS

    for _ in tqdm.tqdm(range(epochs)):
        for unit, a in chatData:
            unit = unit.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(unit, attention_mask=a, labels=unit).loss
            loss.backward()
            optim.step()

        torch.save(model.state_dict(), PT_FILE_PATH)


def infer(inp, model, tokenizer, device) -> str:
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer(inp, return_tensors="pt")

    unit = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)

    output = model.generate(unit, attention_mask=a, max_length=MAX_REQUEST_LENGTH)
    output = tokenizer.decode(output[0])
    return output


def train_default_model():
    device = get_device()

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

    train(chat_data, model, optim, device)
