from AI import API_class, Chat


def chatting(using_bot: API_class.API):
    while True:
        try:
            print(using_bot.get_bot_answer(input("> ")))
        except KeyboardInterrupt:
            print("Bye!")
            return


if __name__ == "__main__":
    bot = API_class.API()

    chatting(bot)
