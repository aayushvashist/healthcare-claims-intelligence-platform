from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

print("Key loaded:", key)
print("Length:", len(key))
print("Starts with:", key[:5])