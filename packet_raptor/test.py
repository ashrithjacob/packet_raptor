import os
import json
import requests
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
import boto3
from pyvis.network import Network
from sklearn.cluster import KMeans
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from sklearn.mixture import GaussianMixture
#import streamlit.components.v1 as components
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_aws import BedrockEmbeddings, BedrockLLM, ChatBedrock
from dotenv import load_dotenv
load_dotenv()


session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

session_state={}
HOSTNAME = "localhost"

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


def load_model():
    print("Loading model...")   
    model = "amazon.titan-embed-text-v1"
    embedding_model = BedrockEmbeddings(region_name=os.getenv("AWS_LOCATION"), model_id=model)
    return embedding_model


# Class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, json_path):
        self.embedding_model = load_model()
        #self.llm = OllamaLLM(
        #    model=session_state["selected_model"], base_url=f"http://{HOSTNAME}:11434"
        #)
        self.llm = ChatBedrock(region_name=os.getenv("AWS_LOCATION"), model_id="meta.llama3-1-70b-instruct-v1:0", model_kwargs=dict(temperature=0.1))
        self.document_cluster_mapping = {}
        self.json_path = json_path
        self.conversation_history = []
        self.load_json()
        self.split_into_chunks()
        self.root_node = None
        self.create_leaf_nodes()
        self.build_tree()  # This will build the tree
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False,
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        print("Splitting text into semantic chunks...")
        self.text_splitter = SemanticChunker(self.embedding_model)
        self.docs = self.text_splitter.split_documents(self.pages)

    def create_leaf_nodes(self):
        print("Creating leaf nodes...")
        self.leaf_nodes = [Node(text=doc.page_content) for doc in self.docs]
        self.embed_leaf_nodes()
        print(f"Leaf nodes created. Total count: {len(self.leaf_nodes)}")

    def embed_leaf_nodes(self):
        for (idx,leaf_node) in enumerate(self.leaf_nodes):
            try:
                print(f"Embedding leaf node {idx}...")
                embedding = self.embedding_model.embed_query(leaf_node.text)
                if embedding is not None and not np.isnan(embedding).any():
                    leaf_node.embedding = embedding
                else:
                    # Handle the case where embedding is nan or None
                    print(
                        f"Invalid embedding generated for leaf node with text: {leaf_node.text}"
                    )
            except Exception as e:
                print(f"Error embedding leaf node: {e}")

    def determine_initial_clusters(self, nodes):
        # This is a simple heuristic: take the square root of the number of nodes,
        # capped at a minimum of 2 and a maximum that makes sense for your application.
        return max(2, int(len(nodes) ** 0.5))

    def cluster_nodes(self, nodes, n_clusters=2):
        print(f"Clustering {len(nodes)} nodes into {n_clusters} clusters...")
        embeddings = np.array(
            [node.embedding for node in nodes if node.embedding is not None]
        )
        print("Embeddings as of Cluster Nodes:", embeddings)
        # Check if embeddings is empty
        if embeddings.size == 0:
            # Handle the case where there are no embeddings to cluster
            # This could be logging a warning and returning the nodes as a single cluster or any other logic you see fit
            print(
                "Warning: No valid embeddings found for clustering. Returning nodes as a single cluster."
            )
            return [nodes]  # Return all nodes as a single cluster to avoid crashing

        # Check if embeddings is not empty but a 1D array, reshape it
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        # Proceed with KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        try:
            kmeans.fit(embeddings)
        except ValueError as e:
            # Handle possible errors during fit, such as an invalid n_clusters value
            print(f"Error during clustering: {e}")
            return [nodes]  # Fallback: return all nodes as a single cluster

        # Initialize clusters based on labels
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, kmeans.labels_):
            clusters[label].append(node)

        print(f"Clusters formed: {len(clusters)}")
        return clusters

    def summarize_cluster(self, cluster):
        # Combine texts from all nodes in the cluster
        combined_text = " ".join([node.text for node in cluster])
        # Generate a summary for the combined text using the LLM
        summary = self.invoke_summary_generation(combined_text)
        summary_embedding = self.embedding_model.embed_query(summary)
        return summary

    def recursive_cluster_summarize(self, nodes, depth=0, n_clusters=None):
        print(
            f"Clustering and summarizing at depth {depth} with {len(nodes)} nodes..."
        )
        if len(nodes) <= 1:
            self.root_node = nodes[0]  # If only one node, it is the root
            return nodes[0]  # Base case: only one node, it is the root

        if n_clusters is None:
            n_clusters = self.determine_initial_clusters(nodes)

        clusters = self.cluster_nodes(nodes, n_clusters=n_clusters)
        parent_nodes = []
        for cluster in clusters:
            cluster_summary = self.summarize_cluster(cluster)
            parent_nodes.append(Node(text=cluster_summary, children=cluster))

        # When we make the recursive call, we don't pass n_clusters, assuming
        # the function will determine the appropriate number for the next level.
        print(f"Clustering and summarization complete at depth {depth}.")
        return self.recursive_cluster_summarize(parent_nodes, depth + 1)

    def build_tree(self):
        print("Building tree...")
        # Determine the number of clusters to start with
        # It could be a function of the number of leaf nodes, or a fixed number
        n_clusters = self.determine_initial_clusters(self.leaf_nodes)
        # Begin recursive clustering and summarization
        self.recursive_cluster_summarize(self.leaf_nodes, n_clusters=n_clusters)
        root_summary_embedding = self.embedding_model.embed_query(self.root_node.text)

    def store_in_chroma(self):
        print("Storing in Chroma")
        all_texts = []
        all_summaries = []

        def traverse_and_collect(node):
            # Base case: if it's a leaf node, collect its text.
            if node.is_leaf():
                all_texts.append(node.text)
            else:
                # For non-leaf nodes, collect the summary.
                all_summaries.append(node.text)
                # Recursively process children.
                for child in node.children:
                    traverse_and_collect(child)

        # Start the traversal from the root node.
        traverse_and_collect(self.root_node)

        # Combine leaf texts and summaries.
        combined_texts = all_texts + all_summaries
        # Now, use all_texts to build the vectorstore with Chroma

        self.vectordb = Chroma.from_texts(
            texts=combined_texts, embedding=self.embedding_model
        )

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def setup_conversation_retrieval_chain(self):
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectordb.as_retriever(search_kwargs={"k": 10}),
            memory=self.memory,
        )

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        if embeddings is None or len(embeddings) == 0:
            print("No reduced embeddings available for clustering.")
            return 1  # Return a default or sensible value for your application
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters + 1)
        ]
        return bics.index(min(bics)) + 1

    def generate_related_queries(self, original_query):
        """
        Create a list of related queries based on the initial question

        :param original_query: Initial question
        :return: Related queries generated by the LLM. If none, empty tuple.
        """
        # NOTE: This prompt is split on sentences for readability. No newlines
        # will be included in the output due to implied line continuation.
        prompt = (
            f"In light of the original inquiry: '{original_query}', let's "
            "delve deeper and broaden our exploration. Please construct a "
            "JSON array containing four distinct but interconnected search "
            "queries. Each query should reinterpret the original prompt's "
            "essence, introducing new dimensions or perspectives to "
            "investigate. Aim for a blend of complexity and specificity in "
            "your rephrasings, ensuring each query unveils different facets "
            "of the original question. This approach is intended to "
            "encapsulate a more comprehensive understanding and generate the "
            "most insightful answers possible. Only respond with the JSON "
            "array itself with a dictionary of 'query' and the new question. "
        )
        response = self.llm.invoke(input=prompt)

        if hasattr(response, "content"):
            # Directly access the 'content' if the response is the expected object
            generated_text = response.content
        elif isinstance(response, dict):
            # Extract 'content' if the response is a dict
            generated_text = response.get("content")
        else:
            # Fallback if the structure is different or unknown
            generated_text = str(response)
            print("Unexpected response format.")

        # st.write("Response content:", generated_text)

        # Assuming the 'content' starts with "content='" and ends with "'"
        # Attempt to directly parse the JSON part, assuming no other wrapping
        related_queries = self.extract_json_from_response(generated_text)

        return related_queries

    def create_synthesis_prompt(self, original_question, all_results):
        """
        Create a prompt based on the original question to gain a composite
        prompt across the highest scored documents.

        :param original_question: Original prompt sent to the LLM
        :param all_results: Sorted (by score) results of original prompt
        :return: Prompt for a composite score based on original_question
        """
        # Sort the results based on RRF score if not already sorted; highest scores first
        prompt = (
            f"Based on the user's original question: '{original_question}', "
            "here are the answers to the original and related questions, "
            "Please synthesize a comprehensive answer focusing on answering the original "
            "question using all the information provided below:\n\n"
            f"{ all_results }"
            "Given the above answers, please provide the best possible composite synthetic answer"
            "to the user's original question."
        )
        return prompt

    def extract_json_from_response(self, response_text):
        """
        If a response is received that should have JSON embedded in the
        output string, look for the opening and closing tags ([]) then extract
        the matching text.

        :param response_text: Response from LLM that might contain JSON
        :return: Python object returned by json.loads. If no JSON response
            was identified, an empty tuple.
        """
        json_result = ()
        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            json_str = response_text[json_start:json_end]
            json_result = json.loads(json_str)
            # st.write("Parsed related queries:", related_queries)
        except (ValueError, json.JSONDecodeError) as e:
            #logger.error("Failed to parse JSON: %s", e)
            print(f"Failed to parse JSON: {e}")
            # related_queries = []
        return json_result

    def chat(self, question):
        print("generating AI multi-query questions")
        # Generate related queries based on the initial question
        related_queries_dicts = self.generate_related_queries(question)

        print("Related Queries:", related_queries_dicts)

        # Ensure that queries are in string format, extracting the 'query' value from dictionaries
        related_queries_list = [q["query"] for q in related_queries_dicts]

        # Combine the original question with the related queries
        queries = [question] + related_queries_list

        all_results = []

        for query_text in queries:
            # response = None
            response = self.qa.invoke(query_text)

            # Process the response
            if response:
                print("Query: ", query_text)
                print("Response: ", response["answer"])
                all_results.append({"query": query_text, "answer": response["answer"]})
            else:
                print("No response received for: ", query_text)

        # After gathering all results, let's ask the LLM to synthesize a comprehensive answer
        if all_results:
            synthesis_prompt = self.create_synthesis_prompt(question, all_results)
            synthesized_response = self.llm.invoke(synthesis_prompt)

            if synthesized_response:
                # Assuming synthesized_response is an AIMessage object with a 'content' attribute
                print("OOOOOO:",synthesized_response.content)
                final_answer = synthesized_response.content
            else:
                final_answer = "Unable to synthesize a response."

            # Update conversation history with the original question and the synthesized answer
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=final_answer))

            return {final_answer}

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content="No answer available."))
        return {"answer": "No results were available to synthesize a response."}

    def update_conversation_history(self, question, response, relevant_nodes):
        # Store the question, response, and the paths through the tree that led to the response
        self.conversation_history.append(
            {"You": question, "AI": response, "Nodes": relevant_nodes}
        )

    def identify_relevant_clusters(self, documents):
        """
        Identify clusters relevant to the given list of documents.

        Parameters:
            documents (list): A list of documents for which to identify relevant clusters.

        Returns:
            set: A set of unique cluster IDs relevant to the given documents.
        """
        cluster_ids = set()
        for i, doc in enumerate(documents):
            cluster_id = self.document_cluster_mapping.get(i)
            if cluster_id is not None:
                cluster_ids.add(cluster_id)
        return cluster_ids

    def get_document_ids_from_clusters(self, clusters):
        # Assuming each cluster (or node) has a list of document IDs associated with it
        document_ids = []
        for cluster in clusters:
            # Assuming `cluster.documents` holds the IDs of documents in this cluster
            document_ids.extend(cluster.documents)
        return document_ids

    def prepare_clustered_data(self, clusters=None):
        """
        Prepare data for summarization by clustering.
        Can now filter by specific clusters if provided, enhancing dynamic use.
        """
        # Initialize a list for filtered documents
        filtered_docs = []

        # If specific clusters are provided, filter documents belonging to those clusters
        if clusters is not None:
            for i, doc in enumerate(self.docs):
                # Retrieve the cluster ID from the mapping using the document's index
                cluster_id = self.document_cluster_mapping.get(i)
                # If the document's cluster ID is in the specified clusters, include the document
                if cluster_id in clusters:
                    filtered_docs.append(doc)
        else:
            filtered_docs = self.docs

        # Construct a DataFrame from the filtered documents
        # You'll need to adjust how you access the text and cluster_id based on your Document object structure
        df = pd.DataFrame(
            {
                "Text": [
                    doc.page_content for doc in filtered_docs
                ],  # Adjust based on your document's structure
                "Cluster": [
                    self.document_cluster_mapping[i] for i in range(len(filtered_docs))
                ],  # Access cluster IDs from the mapping
            }
        )
        self.clustered_texts = self.format_cluster_texts(df)

    def format_cluster_texts(self, df):
        """Organize texts by their clusters."""
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts

    def generate_summaries(self, clusters=None):
        self.prepare_clustered_data(clusters)
        summaries = {}
        for cluster, text in self.clustered_texts.items():
            summary = self.invoke_summary_generation(text)
            summaries[cluster] = summary
        return summaries

    def invoke_summary_generation(self, text):
        """Invoke the language model to generate a summary for the given text."""
        template = "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm.invoke | StrOutputParser()

        summary = chain.invoke({"text": text})
        print("Generated Summary:", summary)
        return summary

    def display_summaries(self, node=None, level=0, summaries=None):
        if summaries is None:
            summaries = []
        if node is None:
            node = self.root_node
        if node is not None:
            # Assuming that node.text contains the summary
            summaries.append((" " * (level * 2)) + node.text)
            for child in node.children:
                self.display_summaries(child, level + 1, summaries)
        return summaries

    def create_tree_graph(self):
        def add_nodes_recursively(graph, node, parent_name=None):
            # Create a unique name for the node
            node_name = f"{node.text[:10]}..." if node.text else "Root"
            graph.add_node(node_name)

            # Link the node to its parent if it exists
            if parent_name:
                graph.add_edge(parent_name, node_name)

            for child in node.children:
                add_nodes_recursively(graph, child, node_name)

        G = nx.DiGraph()
        add_nodes_recursively(G, self.root_node)
        return G


# Class for leaf nodes
class Node:
    def __init__(self, text, children=None, embedding=None):
        self.text = text  # The original text or the summary text of the node
        self.children = children if children is not None else []  # Child nodes
        self.embedding = embedding  # Embedding of the node's text
        self.cluster_label = None  # The cluster this node belongs to

    def is_leaf(self):
        # A leaf node has no children
        return len(self.children) == 0

# temp (just testing)
def extract_and_write_first_n_elements(input_file, output_file, n=100):
    try:
        # Read the original JSON file
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Extract first 100 elements (or all if less than 100)
        first_n_elements = data[:n]
        
        # Write the first 100 elements to a new JSON file
        with open(output_file, 'w') as outfile:
            json.dump(first_n_elements, outfile, indent=2)
        
        print(f"Successfully extracted {len(first_n_elements)} elements to {output_file}")

    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {input_file}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f"tshark -nlr {pcap_path} -T json > {json_path}"
    subprocess.run(command, shell=True)


def get_ollama_models(base_url):
    try:
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()

        # Extract just the model names for the dropdown
        models = [model["name"] for model in models_data.get("models", [])]
        return models
    except requests.exceptions.RequestException as e:
        print(f"Failed to get models from Ollama: {e}")
        return []


# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    print("Packet Raptor - Chat with Packet Captures as a Tree")
    pcap_path = "/home/ash/github/packet_raptor/packet_raptor/rpl-dio-mc-nsa-optional-tlv-dissector-sample.pcap"
    json_path = pcap_path.split('.')[0] + ".json"
    json_new = pcap_path.split('.')[0] + "_new.json"
    pcap_to_json(pcap_path, json_path)
    extract_and_write_first_n_elements(json_path, json_new, n=20)
    session_state["json_path"] = json_path
    session_state["json_new"] = json_new
    print("PCAP file uploaded and converted to JSON.")

    # Fetch and display the models in a select box
    models = get_ollama_models(
        f"http://{HOSTNAME}:11434/"
    )  # Make sure to use the correct base URL
    if models:
        selected_model =  models[0]
        session_state["selected_model"] = selected_model
        print("Selected model:", selected_model)



# Streamlit UI for chat interface
def chat_interface():
    print("Packet Raptor - Chat with Packet Captures as a Tree")
    #json is loaded here
    json_path = session_state["json_path"]
    if not json_path or not os.path.exists(json_path):
        print(
            "PCAP file missing or not converted. Please go back and upload a PCAP file."
        )
        return

    if "chat_instance" not in session_state:
        session_state["chat_instance"] = ChatWithPCAP(json_path=json_path)

    # Visualize the tree
    if session_state["chat_instance"].root_node:
        print("RAPTOR Tree Visualization")
        G = session_state["chat_instance"].create_tree_graph()
        nt = Network("800px", "800px", notebook=True, directed=True)
        nt.from_nx(G)
        nt.hrepulsion(
            node_distance=120,
            central_gravity=0.0,
            spring_length=100,
            spring_strength=0.01,
            damping=0.09,
        )
        nt.set_options(
            """
            var options = {
              "nodes": {
                "scaling": {
                  "label": true
                }
              },
              "edges": {
                "color": {
                  "inherit": true
                },
                "smooth": false
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 300
              },
              "physics": {
                "hierarchicalRepulsion": {
                  "centralGravity": 0.0
                },
                "solver": "hierarchicalRepulsion"
              }
            }
            """
        )
        nt.show("tree.html")
        HtmlFile = open("tree.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        #components.html(source_code, height=800, width=1000)

    user_input = input("Ask a question about the PCAP data:")
    if user_input:
        print("Thinking...")
        response = session_state["chat_instance"].chat(user_input)

        # New: Display the automated AI analysis (summaries) before asking a question
        if session_state["chat_instance"].root_node:
            summaries = session_state["chat_instance"].display_summaries()
            for summary in summaries:
                print(f"* {summary}")
        # Display the top result's answer as markdown for better readability
        if response:
            print("**Sythensized Composite Answer:**")
            # Assuming top_result is a set
            if isinstance(response, set):
                # Convert each element in the set to a string and join them with a line break
                formatted_string = "\n".join(str(element) for element in response)
            print(f"> {formatted_string}")
        else:
            print("No top result available.")

            # Display chat history
            #print("**Chat History:**")
            #for message in session_state["chat_instance"].conversation_history:
            #    # Check the type of message and set the appropriate prefix
            #    prefix = "*User:* " if isinstance(message, HumanMessage) else "*AI:* "
            #    # Access the 'content' attribute of the message to display its text
            #    print(f"{prefix}{message.content}")


if __name__ == "__main__":
    upload_and_convert_pcap()
    chat_interface()
