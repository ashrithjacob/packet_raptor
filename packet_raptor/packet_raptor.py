import os
import umap.umap_ as umap
import pandas as pd
from sklearn.mixture import GaussianMixture
import json
from langchain.load import dumps, loads
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

# Function to generate priming text based on pcap data
def returnSystemText(pcap_data: str) -> str:
    PACKET_WHISPERER = f"""
        You are a helper assistant specialized in analysing packet captures used for troubleshooting & technical analysis. Use the information present in packet_capture_info to answer all the questions truthfully. If the user asks about a specific application layer protocol, use the following hints to inspect the packet_capture_info to answer the question. Format your response in markdown text with line breaks & emojis.

        hints :
        http means tcp.port = 80
        https means tcp.port = 443
        snmp means udp.port = 161 or udp.port = 162
        ntp means udp.port = 123
        ftp means tcp.port = 21
        ssh means tcp.port = 22
        BGP means tcp.port = 179
        OSPF uses IP protocol 89 (not TCP/UDP port-based, but rather directly on top of IP)
        MPLS doesn't use a TCP/UDP port as it's a data-carrying mechanism for high-performance telecommunications networks
        DNS means udp.port = 53 (also tcp.port = 53 for larger queries or zone transfers)s
        DHCP uses udp.port = 67 for the server and udp.port = 68 for the client
        SMTP means tcp.port = 25 (for email sending)
        POP3 means tcp.port = 110 (for email retrieval)
        IMAP means tcp.port = 143 (for email retrieval, with more features than POP3)
        HTTPS means tcp.port = 443 (secure web browsing)
        LDAP means tcp.port = 389 (for accessing and maintaining distributed directory information services over an IP network)
        LDAPS means tcp.port = 636 (secure version of LDAP)
        SIP means tcp.port = 5060 or udp.port = 5060 (for initiating interactive user sessions involving multimedia elements such as video, voice, chat, gaming, etc.)
        RTP (Real-time Transport Protocol) doesn't have a fixed port but is commonly used in conjunction with SIP for the actual data transfer of audio and video streams.
    """
    return PACKET_WHISPERER

# Define a class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, json_path, token_limit=200):
        self.json_path = json_path
        self.token_limit = token_limit
        self.conversation_history = []
        self.load_json()
        self.split_into_chunks()        
        self.embed_texts()
        self.store_in_chroma()        
        self.reduce_dimensions()
        self.cluster_embeddings()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()
        self.priming_text = self.generate_priming_text()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = SemanticChunker(OpenAIEmbeddings())
        self.docs = self.text_splitter.split_documents(self.pages)

    def embed_texts(self):
        self.embedding_model = OpenAIEmbeddings()
        self.embeddings = []

        for idx, doc in enumerate(self.docs):
            try:
                # Directly access the page_content attribute of the Document object
                json_content_str = doc.page_content  # Adjusted based on the assumption that doc has a .page_content attribute

                # Attempt to embed the extracted content
                embedding = self.embedding_model.embed_query(json_content_str)
                if embedding:  # Assuming embedding successfully returns a non-None result
                    self.embeddings.append(embedding)
                else:
                    st.write(f"No embedding generated for document {idx}")
            except Exception as e:
                st.write(f"Error embedding document {idx}: {e}")

        if len(self.embeddings) == 0:
            st.write("Warning: No embeddings were generated.")
        else:
            st.write("Embeddings length:", len(self.embeddings))

    def store_in_chroma(self):
        self.vectordb = Chroma.from_documents(self.docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def reduce_dimensions(self, dim=2, metric="cosine"):
        self.embeddings_reduced = []  # Default to an empty list
        n_embeddings = len(self.embeddings)
        if n_embeddings > 1:
            n_neighbors = max(min(n_embeddings - 1, int((n_embeddings - 1) ** 0.5)), 2)
            try:
                self.embeddings_reduced = umap.UMAP(
                    n_neighbors=n_neighbors, n_components=dim, metric=metric
                ).fit_transform(self.embeddings)
            except Exception as e:
                st.write(f"Error during UMAP dimensionality reduction: {e}")
        else:
            st.write("Not enough embeddings for dimensionality reduction.")

    def cluster_embeddings(self, random_state=0):
        # Check if embeddings_reduced is not None and has a suitable shape for clustering
        if self.embeddings_reduced is not None and self.embeddings_reduced.ndim == 2 and len(self.embeddings_reduced) > 1:
            n_clusters = self.get_optimal_clusters(self.embeddings_reduced)
            gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(self.embeddings_reduced)
            # Proceed with clustering
        else:
            st.write("Reduced embeddings are not available or in incorrect shape for clustering.")
            # Handle the case where clustering cannot proceed, such as setting default values or skipping steps that depend on clustering results.

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        if embeddings is None or len(embeddings) == 0:
            st.write("No reduced embeddings available for clustering.")
            return 1  # Return a default or sensible value for your application
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters + 1)
        ]
        return bics.index(min(bics)) + 1

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def generate_priming_text(self):
        pcap_summary = " ".join([str(page) for page in self.pages[:5]])
        return returnSystemText(pcap_summary)

    def chat(self, question):
        # Combine the original question with the priming text
        primed_question = self.priming_text + "\n\n" + question
        response = self.qa.invoke(primed_question)
        return response

    def prepare_clustered_data(self):
        """Prepare data for summarization by clustering."""
        # Assume self.docs_clusters contains cluster IDs for each document
        # and self.docs contains the original documents with their texts
        df = pd.DataFrame({
            "Text": [doc['text'] for doc in self.docs],
            "Cluster": self.docs_clusters
        })
        self.clustered_texts = self.format_cluster_texts(df)

    def format_cluster_texts(self, df):
        """Organize texts by their clusters."""
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts

    def generate_summaries(self):
        """Generate summaries for each cluster of texts."""
        summaries = {}
        for cluster, text in self.clustered_texts.items():
            summary = self.invoke_summary_generation(text)
            summaries[cluster] = summary
        self.summaries = summaries

    def invoke_summary_generation(self, text):
        """Invoke the language model to generate a summary for the given text."""
        template = "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        summary = chain.invoke({"text": text})
        return summary

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    st.title('Packet Buddy - Chat with Packet Captures')
    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    if uploaded_file:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        pcap_to_json(pcap_path, json_path)
        st.session_state['json_path'] = json_path
        st.success("PCAP file uploaded and converted to JSON.")
        st.button("Proceed to Chat", on_click=lambda: st.session_state.update({"page": 2}))

# Streamlit UI for chat interface
def chat_interface():
    st.title('Packet Buddy - Chat with Packet Captures')
    json_path = st.session_state.get('json_path')
    if not json_path or not os.path.exists(json_path):
        st.error("PCAP file missing or not converted. Please go back and upload a PCAP file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(json_path=json_path)

    user_input = st.text_input("Ask a question about the PCAP data:")
    if user_input and st.button("Send"):
        with st.spinner('Thinking...'):
            response = st.session_state['chat_instance'].chat(user_input)
            st.markdown("**Synthesized Answer:**")
            if isinstance(response, dict) and 'answer' in response:
                st.markdown(response['answer'])
            else:
                st.markdown("No specific answer found.")

            st.markdown("**Chat History:**")
            for message in st.session_state['chat_instance'].conversation_history:
                prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                st.markdown(f"{prefix}{message.content}")

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_convert_pcap()
    elif st.session_state['page'] == 2:
        chat_interface()
