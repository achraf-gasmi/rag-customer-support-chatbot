# test_api.py
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("VERCEL_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

print(f"Base URL: {os.getenv('VERCEL_BASE_URL')}")
print(f"Model: {os.getenv('CHAT_MODEL')}")

response = client.chat.completions.create(
    model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(f"Response: {response.choices[0].message.content}")