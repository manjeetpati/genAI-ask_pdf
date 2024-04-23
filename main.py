from flask import Flask,request
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import signal

import socket
socket.setdefaulttimeout(300) # seconds

app = Flask(__name__)

llm = Ollama(model="llama3")

folder_path = "db"

raw_prompt = PromptTemplate.from_template("""
<s>[INST] You are a technical assistant good at searching documents. if you do not have answer from the provided information, say so.[/INST]</s>
[INST] {input}
            Context:{context}
            Answer: 
[/INST]

""")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,is_separator_regex=False)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/llama3",methods=["POST"])
def llama3():
    prompt = request.get_json()
    print(prompt)
    # return prompt['prompt']
    return llm.invoke(prompt['prompt'])

@app.route("/loadPDF",methods=["POST"])
def loadPDF():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/"+file_name
    file.save(save_file)
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    vector_store  = Chroma.from_documents(documents=chunks,embedding=embedding,persist_directory=folder_path)
    vector_store.persist()
    return ""

@app.route("/ask_pdf",methods=["POST"])
def ask_pdf():
    prompt = request.get_json()
    query = prompt['query']
    print(query)

    vector_store = Chroma(persist_directory=folder_path,embedding_function=embedding)

    retiever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":20,
            "score_threshold":0.1
        }
    )

    document_chain = create_stuff_documents_chain(llm,raw_prompt)
    chain = create_retrieval_chain(retiever,document_chain)
    result = chain.invoke({"input": query})
    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


def start_app():
    app.run(host="0.0.0.0",port=8000,debug=True)

if __name__ == "__main__":
    start_app()