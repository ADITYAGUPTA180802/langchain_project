# pdf_ollama.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE, "Data", "AI.pdf")
assert os.path.exists(pdf_path), f"Put your PDF at {pdf_path}"

docs = PyPDFLoader(pdf_path).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)

emb = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(collection_name="pdf_demo", embedding_function=emb,
                  persist_directory=os.path.join(BASE, "chroma_db"))
vectordb.add_documents(chunks)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    "Use the context to answer. If unknown, say so.\n\nContext:\n{context}\n\nQuestion:{question}"
)
llm = ChatOllama(model="llama3.1", temperature=0)
chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

if __name__ == "__main__":
    print("Ask about your PDF (q to quit).")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() == "q": break
        print("\n" + chain.invoke(q))
