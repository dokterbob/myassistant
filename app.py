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
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager

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


# Memoize query_engine
_query_engine = None


async def get_query_engine():
    global _query_engine
    logger.info("Getting query engine.")

    if not _query_engine:
        Settings.llm = Anthropic(
            model="claude-3-opus-20240229", temperature=0.1, max_tokens=1024
        )
        Settings.embed_model = FastEmbedEmbedding(model_name="thenlper/gte-large")
        Settings.context_window = 200 * 1024

        callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
        Settings.callback_manager = callback_manager

        index = await get_index()

        _query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)

    return _query_engine


@cl.on_chat_start
async def start():
    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

    cl.user_session.set("query_engine", await get_query_engine())


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    assert isinstance(query_engine, RetrieverQueryEngine)

    msg = cl.Message(content="", author="Assistant")

    res = await query_engine.aquery(message.content)
    assert isinstance(res, AsyncStreamingResponse)

    async for token in res.async_response_gen:  # type: ignore
        await msg.stream_token(token)
    await msg.send()
