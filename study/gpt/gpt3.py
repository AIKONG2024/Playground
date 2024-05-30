import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "who is Sam Altman?"}
  ]
)

print(response['choices'][0]['message']['content'])