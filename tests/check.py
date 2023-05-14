import json
import difflib
import statistics
import tqdm


def parse_for_answers(path: str, result: str) -> list:
    pairs = []
    with open(path, 'r') as dataset_file, open(result, 'r') as result_file:

        answers = list(map(str.strip, (list(filter(None, result_file.read().splitlines())))))
        it = iter(answers[1:][::2])
        for data in json.load(dataset_file):
            units = []
            for subtopic in data['subtopics']:
                if subtopic['arguments']:
                    for argument in subtopic['arguments']:
                        units.append(argument['claim'])
                    pairs.append((next(it), units))

    return pairs


def similarity(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def check(dataset: str, results: str) -> float:
    ratio = parse_for_answers(dataset, results)
    average = []

    for i in tqdm.tqdm(ratio):
        a, d = i
        for j in d:
            average.append(similarity(a, j))

    return statistics.fmean(average)
