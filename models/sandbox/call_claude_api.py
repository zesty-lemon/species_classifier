from dotenv import load_dotenv
load_dotenv() #loads the api key from .env file

import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "say meow in one word only",
        }
    ],
)
print(message.content)
