import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Pinecone
# For generating embeddings with OpenAI's embedding model
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever

from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="FireBot", page_icon="🔥🚒")
st.title("🔥🚒 FireBot")



@st.cache_resource(ttl="1h")
def configure_retriever():
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))
    # Create OpenAIEmbeddings object using the provided API key
    embeddings = OpenAIEmbeddings()
    from langchain.retrievers import PineconeHybridSearchRetriever
    from pinecone_text.sparse import BM25Encoder
    # load to your BM25Encoder object
    bm25_encoder = BM25Encoder().load("bm25_values_for_ora_bot.json")
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k=5
    )

    llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    return retriever_from_llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            # source = os.path.basename(doc.metadata["path"])
            self.status.write(f"**Document: {idx}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, input_key='question', output_key='answer')
# memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer')
# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-4-0613", temperature=0, streaming=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True,return_source_documents=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])
        # st.error(response)