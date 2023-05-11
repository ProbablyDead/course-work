import torch
from AI import AILearner
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_pretrained() -> tuple:
    device = AILearner.get_device()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>",
                                  "bos_token": "<startofstring>",
                                  "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>:"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("../sources/model/model_state.pt", device))
    model.eval()

    return model, tokenizer, device


class API:
    def __init__(self):
        self.model, self.tokenizer, self.device = load_pretrained()

    def get_bot_answer(self, request: str) -> str:
        result = AILearner.infer(request, self.model, self.tokenizer, self.device)
        result = result.split("<bot>:")[1]
        result = result.replace("<pad>", '')
        result = result.replace("<endofstring>", '')
        return result
