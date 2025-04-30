from dotenv import load_dotenv
import openai
from openai import OpenAI
import os




def send_request(input, model="gpt-4.1-mini", temperature=0.5, max_tokens=1000):

    # Load the api key
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    # Send the request to the API
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.responses.create(
        model=model,
        input=input,
        temperature=temperature,
        max_output_tokens=max_tokens
        )
    
    return response


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