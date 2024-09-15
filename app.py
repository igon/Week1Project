import chainlit as cl
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = os.getenv("OPENAI_ENDPOINT_URL")    
client = openai.AsyncClient(api_key=openai.api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 500,
}

@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages to send to the LLM
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(
        content="",
    )
    await response_message.send()

    # https://platform.openai.com/docs/guides/chat-completions/response-forma
    stream = await client.chat.completions.create(
        **model_kwargs,
        messages= message_history,
        stream=True,
    )
    
    # https://platform.openai.com/docs/guides/chat-completions/response-format
    async for chunk in stream:
        if chunk := chunk.choices[0].delta.content or "":
            await response_message.stream_token(chunk)

    await response_message.update()

    # Add the LLM response to the message history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)