from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

def train (chatData, model, optim):
    epochs = 10

    for i in tqdm.tqdm(range(epochs)):
        for unit, a in chatData:
            optim.zero_grad()
            loss = model(unit, attention_mask=a, labels=unit).loss
            loss.backward()
            optim.step()

        torch.save(model.state_dict(), "model_state.pt")

def infer(inp):
    inp = "<startofstring> " + inp + " <bot>: "
    inp = tokenizer(inp)

    output = model.generate(**inp)
    output = tokenizer.decode(output[0])
    return output


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token" : "<pad>",
                              "bos_token" : "<startofstring>",
                              "eos_token" : "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

chatData = ChatData("sources/qa_Appliances.json", tokenizer)
chatData = DataLoader(chatData, batch_size=64)

model.train()

optim = Adam(model.parameters())

