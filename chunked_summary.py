import json
import os
import textwrap
import time
from datetime import datetime

import openai


def save_file(text, filename):
    with open(filename, "w", encoding="utf-8") as infile:
        infile.write(text)


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as outfile:
        return outfile.read()


def get_openai_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key is not None:
        openai.api_key = env_key

    openai.api_key = read_file("openai_api_key.txt")


def chat_completion(messages, model="gpt-3.5-turbo", max_retries=10):
    retries = 0

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
            )
            retries = 0
        except Exception as e:
            if retries < max_retries:
                retries += 1
                sleep_time = retries * 1.1
                print(f"Error: {e}")
                print(f"Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                continue
            else:
                raise e

        response_message = completion["choices"][0]["message"]

        response_text = response_message["content"].strip()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        out_text = (
            "PROMPT:\n\n"
            + json.dumps(messages, indent=4)
            + "\n\nRESPONSE:\n\n"
            + json.dumps(response_message, indent=4)
        )
        save_file(out_text, f"gpt_logs/{timestamp}_gpt.txt")

        return response_text


if __name__ == "__main__":
    get_openai_key()

    if not os.path.exists("gpt_logs"):
        os.makedirs("gpt_logs")

    input_text = read_file("input.txt")
    chunks = textwrap.wrap(input_text, 2000)

    print("Number of chunks:", len(chunks))
    input("Press Enter to continue...")
    print("\n\n")

    system_prompt = read_file("prompt.txt")
    result = list()
    count = 0
    for chunk in chunks:
        count += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ]

        completion = chat_completion(messages)
        print("-- COMPLETION --\n\n")
        print(f"{completion}\n\n")

        result.append(completion)

        print(f"Chunk {count} of {len(chunks)} completed\n\n")

        time.sleep(1)

    save_file("\n\n".join(result), "output.txt")
