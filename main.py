from tests.test import test

FILE_PATH = "sources/datasets/discussions_debatepedia.json"
TEST_FILE_PATH = "tests/results.txt"

if __name__ == "__main__":
    tests = test(FILE_PATH)

    with open(TEST_FILE_PATH, "w") as file:
        for request, answer, time_completed in tests:
            file.write(f'{request}: (in {time_completed:.2} sec) \n\t\t{answer}\n\n')
