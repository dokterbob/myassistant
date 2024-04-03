import os
import logging
import sys
import chainlit as cl


from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

logger = logging.getLogger()


def get_index():
    logger.info("Building index.")

    try:
        # Load index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # If data directory doesn't exist, create it with hello world txt
        if not os.path.exists("./data"):
            print("Creating data directory with hello_world.txt")
            os.makedirs("./data")
            with open("./data/hello_world.txt", "w") as f:
                f.write("Hello, world!")

        # Build new index
        documents = SimpleDirectoryReader("./data", recursive=True).load_data(
            show_progress=True
        )
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist()

    return index


def get_query_engine():
    logger.info("Getting query engine.")

    Settings.llm = Anthropic(
        model="claude-3-opus-20240229", temperature=0.1, max_tokens=1024
    )
    Settings.embed_model = FastEmbedEmbedding(model_name="thenlper/gte-large")
    # Settings.context_window = 4096

    callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    Settings.callback_manager = callback_manager

    index = get_index()

    return index.as_query_engine(streaming=True, similarity_top_k=2)


@cl.on_chat_start
async def start():
    query_engine = get_query_engine()

    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    assert isinstance(query_engine, RetrieverQueryEngine)

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
