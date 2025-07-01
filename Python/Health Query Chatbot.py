import requests

# INSERT UR OWN HUGGING FACE API
HF_TOKEN = "hf_..."

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Define system-safe prompt template
def build_prompt(user_input):
    prompt = (
        "You are a helpful and friendly AI medical assistant. You provide general health information only. "
        "You do not prescribe, diagnose, or give emergency advice. If something sounds serious, advise the user to see a doctor.\n\n"
        f"User: {user_input}\nAssistant:"
    )
    return {"inputs": prompt}

# Get response from Mistral via Hugging Face
def get_health_response(user_input):
    data = build_prompt(user_input)
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        print(f"Raw response status: {response.status_code}")
        print("Raw response text:", response.text)  # <--- Add this line
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]['generated_text'].split("Assistant:")[-1].strip()
        elif "error" in result:
            return f"⚠️ API Error: {result['error']}"
        else:
            return "❌ Unexpected response format."
    except Exception as e:
        return f"❌ An error occurred: {e}"

# Interactive chatbot loop
if __name__ == "__main__":
    print("🤖 Mistral Health Chatbot (free via Hugging Face, type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Assistant: Take care! Goodbye.")
            break
        answer = get_health_response(user_input)
        print(f"\nAssistant: {answer}\n")
