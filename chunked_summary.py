import textwrap

from gpt_api_tools import OpenAIConfigHandler


def main() -> None:
    config = OpenAIConfigHandler.load_config("config.json")
    client, params = OpenAIConfigHandler.create_client_and_params(config)

    input_text = ""
    with open("input.txt", "r", encoding="utf-8") as file:
        input_text = file.read()
    chunks = textwrap.wrap(input_text, 4096)

    print("Number of chunks:", len(chunks))
    input("Press Enter to continue...")
    print("\n\n")

    system_prompt = ""
    with open("prompt.txt", "r", encoding="utf-8") as file:
        system_prompt = file.read()

    results = []
    for count, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": system_prompt.replace("<<TEXT TO SUMMARIZE>>", chunk),
            }
        ]
        params.messages = messages
        completion = client.chat_completion(params)
        print("-- COMPLETION --\n\n")
        print(f"{completion}\n\n")
        results.append(completion)
        print(f"Chunk {count} of {len(chunks)} completed\n\n")

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write("\n\n".join(results))


if __name__ == "__main__":
    main()
