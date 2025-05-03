from dotenv import load_dotenv
from openai import OpenAI
import os


def send_request(input, model="gpt-4.1-mini", temperature=0.5, max_tokens=1000):

    # Load the api key
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    if OPENAI_KEY is None:
        return 401, None

    # Send the request to the API
    client = OpenAI(api_key=OPENAI_KEY)
    try:
        response = client.responses.create(
            model=model,
            input=input,
            temperature=temperature,
            max_output_tokens=max_tokens
            )
        if response.status == "completed":
            return 200, response
        else:
            return 500, None
    except Exception:
        return 500, None


def build_prompt(prompt, instructions):

    input = [
        {
            "role": "system",
            "content": instructions
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return input