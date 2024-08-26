from openai import OpenAI
import json
import utils.gpt.message_utils as message_utils
from config.gpt_config import GPTMessageField, GPTMessageRole

ROLE = GPTMessageField.ROLE.value
CONTENT = GPTMessageField.CONTENT.value
SYSTEM = GPTMessageRole.SYSTEM.value
USER = GPTMessageRole.USER.value
ASSISTANT = GPTMessageRole.ASSISTANT.value

class GPTClient:
    def __init__(self, config):
        self.model = config.get("model")
        self.open_ai_client = OpenAI(api_key=config.get("api_key"))
        self.messages = []

        print("GPT Client initialized!")
        print(f"Set model is: {self.model}")
    
    def __log_invalid_message(self, message, additional_log):
        print("The following is an invalid message:")
        print(json.dumps(obj=message, indent=4))
        if additional_log:
            print(additional_log)

    def __is_valid_message(self, message, invalid_message_log):
        is_valid = message_utils.is_valid_message(message)
        if not is_valid and invalid_message_log:
            self.__log_invalid_message(
                message=message,
                additional_log=invalid_message_log
            )
        return is_valid
    
    def __are_valid_messages(self, messages, invalid_message_log):
        if not isinstance(messages, (list, tuple)):
            return False
        for message in messages:
            if not self.__is_valid_message(
                message=message,
                invalid_message_log=invalid_message_log
            ):
                return False
        return True
    
    def get_message_history(self):
        return self.messages

    def set_messages(self, messages):
        if self.__are_valid_messages(
            messages=messages,
            invalid_message_log="ABORT! Messages will NOT be overwritten."
        ):
            self.messages = messages
        return self.messages
    
    def add_messages(self, messages):
        if self.__are_valid_messages(
            messages=messages,
            invalid_message_log="ABORT! Messages will NOT be added."
        ):
            self.messages.extend(messages)
        return self.messages
    
    def set_persona_context(self, context_text="You are a helpful assistant"):
        context_message = message_utils.build_context_message(context_text)
        if self.__is_valid_message(
            message=context_message,
            invalid_message_log="ABORT! The new context will NOT be set."
        ):
            self.messages.append(context_message)
        return self.messages
    
    def add_message(self, role, message_content):
        message = message_utils.build_message(role=role, content=message_content)
        if self.__is_valid_message(
            message=message,
            invalid_message_log=f"ABORT! The {role} message will NOT be added."
        ):
            self.messages.append(message)
        return self.messages
    
    def add_user_message(self, message_content):
        user_message = message_utils.build_user_message(message_content)
        if self.__is_valid_message(
            message=user_message,
            invalid_message_log="ABORT! The user message will NOT be added."
        ):
            self.messages.append(user_message)
        return self.messages
    
    def add_user_message_under_new_context(self, message_content, context_text="You are a helpful assistant"):
        messages = message_utils.build_user_messages_with_context(
            user_content=message_content,
            context_text=context_text
        )
        if self.__are_valid_messages(
            messages=messages,
            invalid_message_log="ABORT! New context and user message will NOT be added."
        ):
            self.messages.extend(messages)
        return self.messages
    
    def prompt_model(self, log_completion=False):
        if not self.messages:
            print("No messages set. Aborting prompt operation.")
            return None
        completion_response = self.open_ai_client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        chosen_completion = completion_response.choices[0].message
        last_message = self.messages[-1]
        if chosen_completion:
            current_messages = self.add_message(
                role=chosen_completion.role,
                message_content=chosen_completion.content
            )
            was_response_message_added = current_messages[-1] != last_message
            if was_response_message_added and log_completion:
                print(f"RESPONSE CONTENT: {chosen_completion.content}")
        return completion_response
