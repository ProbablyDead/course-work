import json

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


def test(path: str) -> list:
    bot = API()
    units = parse(path)
    result = []

    for i in units:
        result.append((i, bot.get_bot_answer(i)))

    return result
