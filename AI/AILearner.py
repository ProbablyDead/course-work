from transformers import GPT2LMHeadModel, GPT2Tokenizer
from AI import Chat
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

PT_FILE_PATH = "model_state.pt"
PRETRAINED_MODEL = "gpt2"
DATASET_PATH = "sources/datasets/humor_funny.json"
DEFAULT_MESSAGE = "tell me an joke"
EPOCHS = 10
UNITS_TO_LEARN = 10000
MAX_REQUEST_LENGTH = 40


def get_device () -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(chatData, model, optim, device):
    epochs = EPOCHS

    for i in tqdm.tqdm(range(epochs)):
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

    chatData = Chat(DATASET_PATH, tokenizer)
    chatData = DataLoader(chatData, batch_size=64)

    model.train()

    optim = Adam(model.parameters(), lr=1e-3)

    train(chatData, model, optim, device)
