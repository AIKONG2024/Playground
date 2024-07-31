import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

user_prompt = "오늘 서울의 날씨를 알려주겠어?"
completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0613:personal::9qsygcUp",
    messages = [
    {"role": "system", "content": "냉랭이는 사실적이면서도 풍자적인 챗봇입니다."},
    {"role": "user", "content": user_prompt},
  ]
)

print(completion.choices[0].message)