import chainlit as cl
import openai
import os
from prompts import SYSTEM_PROMPT
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = os.getenv("OPENAI_ENDPOINT_URL")    
client = wrap_openai(openai.AsyncClient(
    api_key=openai.api_key,
    base_url=endpoint_url)
)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 500,
}

ENABLE_SYSTEM_PROMPT = True


@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages to send to the LLM
    message_history = cl.user_session.get("message_history", [])

    if ENABLE_SYSTEM_PROMPT and (not message_history or message_history[0]["role"] != "system"):
        system_prompt_content = SYSTEM_PROMPT
        message_history.insert(0, {"role": "system", "content": system_prompt_content}) 

    #add user message to the history
    message_history.append({"role": "user", "content": message.content})

    # https://platform.openai.com/docs/guides/chat-completions/response-forma
    stream = await client.chat.completions.create(
        **model_kwargs,
        messages= message_history,
        stream=True,
    )
    
    response_message = cl.Message(
    content="", 
    )

    # https://platform.openai.com/docs/guides/chat-completions/response-format
    async for chunk in stream:
        if chunk := chunk.choices[0].delta.content or "":
            await response_message.stream_token(chunk)

    await response_message.update()

    # Add the LLM response to the message history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)



