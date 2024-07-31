import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

file_response = client.files.create(
    file=open('mydata.jsonl', 'rb'),
    purpose='fine-tune'
)

file_id = file_response.id

job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-3.5-turbo-0613",
    hyperparameters={
    "n_epochs":2
  }
)

print(job)
