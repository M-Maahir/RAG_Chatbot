import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


DATA_DIR = "data"
DB_DIR = "db"


def load_documents():
    docs = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        

        elif filename.endswith(".txt") or filename.endswith(".md"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    return docs


def main():
    documents = load_documents()

    if not documents:
        print("No documents found in the data folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectorstore.persist()
    print(f"Indexed {len(chunks)} chunks into '{DB_DIR}'")


if __name__ == "__main__":
    main()
    
    