from torch.utils.data import Dataset
import json

class Chat(Dataset):

    def __init__(self, path:str, tokenizer) -> None:

        self.data = json.load(open(path, 'r'))

        self.units = []
        for i in self.data:
            self.units.append(i['text'])

        for index, i in enumerate(self.units):
            try:
                self.units[index] = "<startofstring> " + i + " <bot>: " + self.units[index+1] + " <endofstring>"
            except:
                pass

        self.units = self.units[:1000]

        print(self.units[0])

        self.units_encoded = tokenizer(self.units, truncation=True, padding=True, return_tensors="pt")
        self.input_ids = self.units_encoded['input_ids']
        self.attention_mask = self.units_encoded['attention_mask']

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index):
        return (self.input_ids[index], self.attention_mask[index])
