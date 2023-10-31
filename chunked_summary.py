import logging
import textwrap
from typing import List, Tuple

from gpt_api_tools import OpenAIConfigHandler
from gpt_api_tools.ChatCompletionParams import ChatCompletionParams
from gpt_api_tools.io_utils import safe_read_text, safe_write_text
from gpt_api_tools.OpenAIClient import OpenAIClient


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("chunked_summary.log", mode="w"),
        ],
    )


def load_config_and_create_client() -> Tuple[OpenAIClient, ChatCompletionParams]:
    config = OpenAIConfigHandler.load_config("config.json")
    return OpenAIConfigHandler.create_client_and_params(config)


def chunk_text(input_text: str) -> List[str]:
    return textwrap.wrap(input_text, 4096)


def summarize_text(
    client: OpenAIClient,
    params: ChatCompletionParams,
    chunks: List[str],
    system_prompt: str,
) -> List[str]:
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
        logging.info("-- COMPLETION --\n\n")
        logging.info(f"{completion}\n\n")
        results.append(completion)
        logging.info(f"Chunk {count} of {len(chunks)} completed\n\n")
    result = "\n\n".join(results)
    return result


def main() -> None:
    setup_logging()

    client, params = load_config_and_create_client()

    input_text = safe_read_text("input.txt")
    if not input_text:
        logging.error("Input file is empty. Exiting...")
        return

    chunks = chunk_text(input_text)
    logging.info(f"Number of chunks: {len(chunks)}")
    input("Press Enter to continue...")

    system_prompt = safe_read_text("prompt.txt")
    if not system_prompt:
        logging.error("Prompt file is empty. Exiting...")
        return

    results = summarize_text(client, params, chunks, system_prompt)

    if not safe_write_text("output.txt", results):
        logging.error("Failed to write output to file. Exiting...")
        return


if __name__ == "__main__":
    main()
