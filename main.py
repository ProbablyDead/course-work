from tests.test import test
from tests.check import check

FILE_PATH = "sources/datasets/discussions_debatepedia.json"
RESULT_FILE_PATH = "tests/results.txt"

if __name__ == "__main__":
    tests = test(FILE_PATH)

    with open(RESULT_FILE_PATH, "w") as file:
        for request, answer, time_completed in tests:
            file.write(f'{request}: (in {time_completed:.2} sec) \n\t\t{answer}\n\n')

        file.write(f"\n\nAverage ratio = {int(check(FILE_PATH, RESULT_FILE_PATH) * 100)}%")
