from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DB_DIR = "db"


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful RAG assistant.

Use the provided context to answer the user's question.
If the user asks a broad question like "what is this about?",
give a short summary of the document content.

for every answer redirect to the part of the context that
answer the questions"

Context:
{context}

Question:
{question}

Answer:
""".strip()


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="deepseek-r1:7b")

    print("RAG Bot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        docs = retriever.invoke(question)   

        if not docs:
            print("Bot: I couldn't find that in the documents.\n")
            continue

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(context, question)

        answer = llm.invoke(prompt)

        print(f"\nBot: {answer}\n")

        print("Sources:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"  {i}. {source} (page: {page})")
        print()


if __name__ == "__main__":
    main()