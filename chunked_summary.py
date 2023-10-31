import json
import os
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import openai
from openai.error import OpenAIError


@dataclass
class ChatCompletionParams:
    messages: List[Dict[str, str]]
    model: str = "gpt-3.5-turbo"
    frequency_penalty: Optional[float] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Any]] = None
    logit_bias: Optional[Dict[int, float]] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or self._load_api_key()
        openai.api_key = self.api_key
        self.ensure_logs_directory_exists()

    @staticmethod
    def _load_api_key() -> str:
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key is not None:
            return env_key
        with open("openai_api_key.txt", "r", encoding="utf-8") as file:
            return file.read().strip()

    @staticmethod
    def ensure_logs_directory_exists() -> None:
        os.makedirs("gpt_logs", exist_ok=True)

    def chat_completion(
        self, params: ChatCompletionParams, max_retries: int = 10
    ) -> str:
        for retry in range(max_retries + 1):
            try:
                response = openai.ChatCompletion.create(**params.to_dict())
                content = response["choices"][0]["message"]["content"].strip()
                self.log_response(params.messages, response)
                return content
            except OpenAIError as e:
                print(f"OpenAI API error: {e}")
                if retry < max_retries:
                    sleep_time = (retry + 1) * 1.1
                    print(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                    time.sleep(sleep_time)
                else:
                    print("Maximum retries reached. Unable to complete the request.")
                    raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

    @staticmethod
    def log_response(messages: List[Dict[str, str]], response: Dict) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_text = json.dumps({"PROMPT": messages, "RESPONSE": response}, indent=4)
        with open(f"gpt_logs/{timestamp}_gpt.txt", "w", encoding="utf-8") as file:
            file.write(out_text)


def main() -> None:
    client = OpenAIClient()

    input_text = ""
    with open("input.txt", "r", encoding="utf-8") as file:
        input_text = file.read()
    chunks = textwrap.wrap(input_text, 2000)

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
        params = ChatCompletionParams(messages=messages)
        completion = client.chat_completion(params)
        print("-- COMPLETION --\n\n")
        print(f"{completion}\n\n")
        results.append(completion)
        print(f"Chunk {count} of {len(chunks)} completed\n\n")

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write("\n\n".join(results))


if __name__ == "__main__":
    main()
