from torch.utils.data import Dataset
import json

NUMBER_OF_UNITS_TO_LEARN = 1000  # -1 is for whole dataset
MAX_BOT_ANSWER_LENGTH = 80


def format_string(q: str, a: str) -> str:
    """
    Formats the question and answer into a string format for training.

    Args:
        q (str): Question string.
        a (str): Answer string.

    Returns:
        str: Formatted string.
    """
    return "<startofstring> " + q[q.find(':') + 2:] + " <bot>: " + a + " <endofstring>"


def parse(path: str, formatter) -> list:
    """
    Parses the dataset file and returns a list of formatted units.

    Args:
        path (str): Path to the dataset file.
        formatter (function): Formatter function to format each unit.

    Returns:
        list: List of formatted units.
    """

    try:
        file = json.load(open(path, 'r'))
    except FileNotFoundError:
        return []

    units = []
    for data in file:
        for subtopic in data['subtopics']:
            question = subtopic['title']
            for argument in subtopic['arguments']:
                units.append(formatter(question, argument['claim']))

    return units


class Chat(Dataset):

    def __init__(self, path: str, tokenizer) -> None:
        """
        Custom Dataset class for training the chatbot.

        Args:
            path (str): Path to the dataset file.
            tokenizer: Tokenizer object.

        Returns:
            None
        """
        self.units = parse(path, format_string)[:NUMBER_OF_UNITS_TO_LEARN]

        self.units_encoded = tokenizer(self.units, max_length=MAX_BOT_ANSWER_LENGTH, truncation=True,
                                       padding="max_length", return_tensors="pt")
        self.input_ids = self.units_encoded['input_ids']
        self.attention_mask = self.units_encoded['attention_mask']

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.units)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing input_ids and attention_mask tensors.
        """
        return self.input_ids[index], self.attention_mask[index]
