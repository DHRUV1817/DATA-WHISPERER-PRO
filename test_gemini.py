import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load env
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

# 2. Basic checks
print("GEMINI_API_KEY found:", bool(key))
if not key:
    exit("❌ Key missing – fix .env file")

# 3. Configure SDK
genai.configure(api_key=key)

# 4. Quick call
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content("Say 'OK' if you are alive.", generation_config={"max_output_tokens": 5})
    print("✅ Gemini OK – response:", resp.text.strip())
except Exception as e:
    print("❌ Gemini error:", e)