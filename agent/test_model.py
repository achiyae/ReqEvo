
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env explicitly
load_dotenv(override=True)

api_key = os.environ.get("OPENAI_API_KEY")
model_name = "gpt-5-nano"

if not api_key:
    print("Error: OPENAI_API_KEY not found in environment.")
    sys.exit(1)

print(f"Testing connectivity to OpenAI with model '{model_name}'...")
print(f"Key used: {api_key[:8]}...{api_key[-4:]}")

try:
    llm = ChatOpenAI(temperature=0, model=model_name, api_key=api_key)
    res = llm.invoke("Hello, simple test.")
    print("\n--- Model Response ---")
    print(res.content)
    print("--- Success ---")
except Exception as e:
    print("\n--- Connection Failed ---")
    print(str(e))
