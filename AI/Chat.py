from torch.utils.data import Dataset
import json

NUMBER_OF_UNITS_TO_LEARN = 1000  # -1 is for whole dataset
MAX_BOT_ANSWER_LENGTH = 80


def parse(path: str) -> list:
    def format_string(q: str, a: str) -> str:
        return "<startofstring> " + q[q.find(':') + 2:] + " <bot>: " + a + " <endofstring>"

    try:
        file = json.load(open(path, 'r'))
    except FileNotFoundError:
        return []

    units = []
    for data in file:
        for subtopic in data['subtopics']:
            question = subtopic['title']
            for argument in subtopic['arguments']:
                units.append(format_string(question, argument['claim']))

    return units


class Chat(Dataset):

    def __init__(self, path: str, tokenizer) -> None:
        self.units = parse(path)[:NUMBER_OF_UNITS_TO_LEARN]

        print(self.units[0])

        self.units_encoded = tokenizer(self.units, max_length=MAX_BOT_ANSWER_LENGTH, truncation=True,
                                       padding="max_length", return_tensors="pt")
        self.input_ids = self.units_encoded['input_ids']
        self.attention_mask = self.units_encoded['attention_mask']

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> tuple:
        return self.input_ids[index], self.attention_mask[index]
