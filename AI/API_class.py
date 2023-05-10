from AI import AILearner
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_pretrained() -> tuple:
    return GPT2LMHeadModel.from_pretrained("../sources/model/pretrained"), \
           GPT2Tokenizer.from_pretrained("../sources/model/pretrained"), \
           AILearner.get_device()


class API:
    def __init__(self):
        self.model, self.tokenizer, self.device = load_pretrained()

    def get_bot_answer(self, request: str) -> str:
        result = AILearner.infer(request, self.model, self.tokenizer, self.device)
        result = result.split("<bot>:")[1]
        result = result.replace("<pad>", '')
        # result = result.replace("<endofstring>", '')
        return result
