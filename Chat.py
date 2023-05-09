from torch.utils.data import Dataset
import json

NUMBER_OF_UNITS_TO_LEARN = 1000  # -1 is for whole dataset
MAX_BOT_ANSWER_LENGTH = 80


class Chat(Dataset):

    def __init__(self, path: str, tokenizer) -> None:

        self.data = json.load(open(path, 'r'))

        self.units = []
        for i in self.data:
            self.units.append(i['text'])

        for index, i in enumerate(self.units):
            try:
                self.units[index] = "<startofstring> " + i + " <bot>: " + self.units[index + 1] + " <endofstring>"
            except:
                pass

        self.units = self.units[:NUMBER_OF_UNITS_TO_LEARN]

        print(self.units[0])

        self.units_encoded = tokenizer(self.units, max_length=MAX_BOT_ANSWER_LENGTH, truncation=True,
                                       padding="max_length", return_tensors="pt")
        self.input_ids = self.units_encoded['input_ids']
        self.attention_mask = self.units_encoded['attention_mask']

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index):

        return self.input_ids[index], self.attention_mask[index]
