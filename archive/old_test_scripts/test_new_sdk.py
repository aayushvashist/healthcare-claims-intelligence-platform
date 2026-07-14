from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("Loaded key:", api_key[:10], "...")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello in one sentence."
)

print(response.text)
print(api_key)
print(len(api_key))