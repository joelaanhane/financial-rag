from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": "What are the most important things to look for in an annual report?"}
    ]
)

print(response.choices[0].message.content)