# Chunked-Summary

Chunked-Summary is a Python utility for processing large text data in chunks and generating a summary or response using OpenAI's GPT-3.5-turbo API. This utility is useful when the input text is too long to be processed by the API in a single request.

## Requirements

To use this utility, you need to have Python installed on your machine. You also need an API key from OpenAI. The key can be provided as an environment variable (`OPENAI_API_KEY`) or stored in a file named `openai_api_key.txt`.

## Installation

1. Clone the repository to your local machine.
2. Install the required packages:
   
   ```
   pip install openai
   ```

## Usage

1. Place your input text in a file named `input.txt`.
2. (Optional) If you have a custom prompt or instruction for the model, place it in the file named `prompt.txt`.
3. Run the script:

   ```
   python chunked_summary.py
   ```

4. The script will display the number of chunks that the input text has been divided into and will wait for you to press Enter before continuing.
5. The script will then process each chunk sequentially, displaying the output for each and logging the interaction to the `gpt_logs` directory.
6. Once all chunks have been processed, the combined output will be saved to `output.txt`.

## Configuration

The `ChatCompletionParams` class in the script can be modified to include additional parameters for the OpenAI API call, such as temperature, max tokens, etc.

## Logs

All interactions with the OpenAI API are logged in the `gpt_logs` directory. Each log file is named with a timestamp and contains both the input messages and the API response.

## Error Handling

The script includes basic error handling for OpenAI API errors, with exponential backoff retry logic. The maximum number of retries can be configured via the `max_retries` parameter in the `chat_completion` method.

## License

This project is released under the MIT License.

---

**Note**: This script interacts with OpenAI's GPT-3.5-turbo API, which may incur costs depending on your usage. Ensure that you have reviewed OpenAI's pricing details and are aware of the costs associated with using their API.

Enjoy generating summaries and responses with Chunked-Summary!
