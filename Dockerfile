FROM python:3

RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl

RUN apt-get update

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /usr/src/app

RUN pip install -U pip
RUN pip install spacy

COPY . .

RUN pip install -r requirements.txt
RUN pip install pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl


CMD [ "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
