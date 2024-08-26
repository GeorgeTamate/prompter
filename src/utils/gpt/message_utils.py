from config.gpt_config import GPTMessageField, GPTMessageRole

def is_message_a_dictionary(message):
    return isinstance(message, dict)

def has_required_message_fields(message):
    if not is_message_a_dictionary(message):
        return False
    for key in message:
        if key not in (field.value for field in GPTMessageField):
            return False
    return True

def has_valid_message_role(message):
    if not is_message_a_dictionary(message):
        return False
    message_role = message.get(GPTMessageField.ROLE.value)
    if not message_role:
        return False
    if message_role not in (role.value for role in GPTMessageRole):
        return False
    return True

def has_message_content(message):
    return (
        is_message_a_dictionary(message)
        and bool(message.get(GPTMessageField.CONTENT.value))
    )

def is_valid_message(message):
    if (
        has_required_message_fields(message)
        and has_valid_message_role(message)
        and has_message_content(message)
    ):
        return True
    return False

def build_message(role, content):
    return {
        GPTMessageField.ROLE.value: role,
        GPTMessageField.CONTENT.value: content,
    }

def build_context_message(context_text):
    return build_message(role=GPTMessageRole.SYSTEM.value, content=context_text)

def build_user_message(content):
    return build_message(role=GPTMessageRole.USER.value, content=content)

def build_assistant_message(content):
    return build_message(role=GPTMessageRole.ASSISTANT.value, content=content)

def build_messages_with_context(role, content, context_text):
    context_message = build_context_message(context_text)
    message = build_message(role=role, content=content)
    return (
        context_message,
        message,
    )     

def build_user_messages_with_context(user_content, context_text):
    return build_messages_with_context(
        role=GPTMessageRole.USER.value,
        content=user_content,
        context_text=context_text
    )
