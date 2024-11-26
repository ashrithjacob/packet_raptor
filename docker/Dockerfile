FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install --break-system-packages -U --quiet langchain==0.3.7\
  && pip install --break-system-packages -U --quiet langchain_experimental==0.3.3\
  && pip install --break-system-packages -U --quiet langchain-community==0.3.7\
  && pip install --break-system-packages langchain-openai==0.2.9\
  && pip install --break-system-packages langchain-ollama==0.2.0 \
  && pip install --break-system-packages chromadb==0.5.20 \
  && pip install --break-system-packages langchain-aws==0.2.7 \
  && pip install --break-system-packages chromadb==0.5.20\
  && pip install --break-system-packages tiktoken

RUN echo "==> Install boto3..." \
  && pip install --break-system-packages boto3==1.35.69

RUN echo "==> Install jq.." \
  && pip install --break-system-packages jq

RUN echo "==> Install streamlit.." \
  && pip install --break-system-packages streamlit --upgrade

# Install tshark
RUN apt-get update && apt-get install -y tshark

RUN echo "==> Adding pyshark ..." \
  && pip install --break-system-packages pyshark

RUN echo "==> Adding pandas ..." \
  && pip install --break-system-packages pandas

RUN echo "==> Adding scikit-learn ..." \
  && pip install --break-system-packages scikit-learn==1.5.2

RUN echo "==> Adding networkx ..." \
  && pip install --break-system-packages networkx

RUN echo "==> Adding pyvis ..." \
  && pip install --break-system-packages pyvis==0.3.2

RUN echo "==> Adding InstructorEmbedding ..." \
  && pip install --break-system-packages -U sentence-transformers==2.2.2 \
  && pip install --break-system-packages InstructorEmbedding

COPY /packet_raptor /packet_raptor/
COPY /scripts /scripts/
COPY .env .env

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]