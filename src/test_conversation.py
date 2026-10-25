
from src.conversation import ConversationController


def main():
    agent = ConversationController()

    print("ZentroMart support assistant")
    print("Type 'exit' to stop.\n")

    while True:
        message = input("Customer: ").strip()

        if message.lower() in {"exit", "quit"}:
            break

        try:
            reply = agent.handle_message(message)
            print(f"Assistant: {reply}\n")
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
