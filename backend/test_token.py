import os
import requests
from dotenv import load_dotenv

load_dotenv()  # loads your .env file

token = os.getenv("GITHUB_TOKEN")
if not token:
    print("❌ GITHUB_TOKEN not found in environment.")
    exit(1)

url = "https://api.github.com/user"
headers = {"Authorization": f"token {token}"}

resp = requests.get(url, headers=headers)

if resp.status_code == 200:
    data = resp.json()
    print("✅ Token works! Authenticated as:", data["login"])
else:
    print(f"❌ Token failed with status {resp.status_code}: {resp.text}")
