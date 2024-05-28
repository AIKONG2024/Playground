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
    {"role": "user", "content": "샘 알트먼은 누구인가요? 그의 경력과 업적에 대해 알려주세요."}
  ]
)

print(response['choices'][0]['message']['content'])