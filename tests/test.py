from alive_progress import alive_bar
import json
import time

from AI.API_class import API


def parse(path: str) -> list:
    try:
        file = json.load(open(path, 'r'))
    except FileNotFoundError:
        return []

    units = []
    for data in file:
        for subtopic in data['subtopics']:
            if subtopic["arguments"]:
                units.append(subtopic["title"][subtopic["title"].find(':') + 2:])

    return units


def chatting(using_bot: API) -> None:
    while True:
        try:
            print(using_bot.get_bot_answer(input("> ")))
        except KeyboardInterrupt:
            print("Bye!")
            return


def test(path: str) -> tuple:
    bot = API()
    units = parse(path)
    result = []
    start_time = time.time()

    with alive_bar(len(units)) as bar:
        for i in units:
            bar()
            start_time = time.time()
            answer = bot.get_bot_answer(i)
            result.append((i, answer, time.time() - start_time))

    return result
