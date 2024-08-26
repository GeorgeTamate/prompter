from dotenv import load_dotenv
import json
from config.gpt_config import get_gpt_config
from helpers.GPTClient import GPTClient

# Load environment variables from .env file
load_dotenv()

try:
    # Load custom client config
    gpt_client_config = get_gpt_config()

    # Intance custom client
    gpt_client = GPTClient(gpt_client_config)

    # Simulate failed prompting
    gpt_client.prompt_model()

    # Add initial messages and system context to history
    messages = gpt_client.add_user_message_under_new_context(
        context_text="You are a helpful assistant that responds only in Spanish regardless the input language.",
        message_content="Describe the city of New York in 3 sentences."
    )

    # Prompting with actual messages (also prints received message from API)
    gpt_client.prompt_model(log_completion=True)

    # Display resulting history
    message_history = gpt_client.get_message_history()
    print("MESSAGE HISTORY:")
    print(json.dumps(obj=message_history, indent=4))

except ValueError as e:
    print(f"ERROR: {e}")
