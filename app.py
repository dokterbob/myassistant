import os
import logging
import chainlit as cl


from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.agent import AgentChatResponse
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer

logger = logging.getLogger()


async def get_index():
    documents = SimpleDirectoryReader(
        "./data", recursive=True, filename_as_id=True
    ).aload_data()

    try:
        logger.info("Loading existing index from 'storage'.")

        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        index.refresh_ref_docs(await documents)
    except FileNotFoundError:
        logger.info("Building new index from 'data'.")

        # If data directory doesn't exist, create it with hello world txt
        if not os.path.exists("./data"):
            print("Creating data directory with hello_world.txt")
            os.makedirs("./data")
            with open("./data/hello_world.txt", "w") as f:
                f.write("Hello, world!")

        index = VectorStoreIndex(await documents, show_progress=True, use_async=True)

    index.storage_context.persist()

    return index


# Memoize chat_engine
_chat_engine = None


async def get_chat_engine():
    global _chat_engine
    logger.info("Getting query engine.")

    if not _chat_engine:
        Settings.llm = Anthropic(
            model="claude-3-opus-20240229", temperature=0.1, max_tokens=4096
        )
        Settings.embed_model = FastEmbedEmbedding(model_name="thenlper/gte-large")
        Settings.context_window = 200 * 1024

        callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
        Settings.callback_manager = callback_manager

        index = await get_index()

        memory = ChatMemoryBuffer.from_defaults(token_limit=10 * 1024)

        _chat_engine = index.as_chat_engine(
            similarity_top_k=10, memory=memory, streaming=True
        )

    return _chat_engine


@cl.on_chat_start
async def start():
    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

    cl.user_session.set("chat_engine", await get_chat_engine())


@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    assert isinstance(chat_engine, BaseChatEngine)

    msg = cl.Message(content="", author="Assistant")

    res = await chat_engine.astream_chat(message.content)
    assert isinstance(res, StreamingAgentChatResponse)

    async for token in res.async_response_gen():
        await msg.stream_token(token)
    await msg.send()
