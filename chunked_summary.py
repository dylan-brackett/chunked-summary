import logging
import textwrap

from gpt_api_tools import OpenAIConfigHandler
from gpt_api_tools.io_utils import safe_read_text, safe_write_json


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("chunked_summary.log", mode="w"),
        ],
    )

    config = OpenAIConfigHandler.load_config("config.json")
    client, params = OpenAIConfigHandler.create_client_and_params(config)

    input_text = safe_read_text("input.txt")
    if not input_text:
        logging.error("Input file is empty. Exiting...")
        return

    chunks = textwrap.wrap(input_text, 4096)

    logging.info("Number of chunks:", len(chunks))
    input("Press Enter to continue...")
    logging.info("\n\n")

    system_prompt = safe_read_text("prompt.txt")
    if not system_prompt:
        logging.error("Prompt file is empty. Exiting...")
        return

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

    if not safe_write_json("output.json", results):
        logging.error("Failed to write output to file. Exiting...")
        return


if __name__ == "__main__":
    main()
