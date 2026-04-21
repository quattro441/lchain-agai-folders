import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeSparseVectorStore, PineconeVectorStore
# OLLAMAembeddings funkade inte...
from langchain_pinecone.embeddings import PineconeSparseEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

print(f"Initializing components...")


# MODEL = "qwen3:14b"
MODEL = "gemma4:e4b"

embeddings = OllamaEmbeddings(model="qwen3-embedding", dimensions=1536)
# embeddings = PineconeSparseEmbeddings()
llm = ChatOllama(model=MODEL)

# vectorstore = PineconeSparseVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
        Answer the question only based on the following context:
        {context}

        Question: {question}

        Provide a detailed answer: 
    """
)

# Auxillary function for formatting text
def format_docs(docs):
    """
        docs - langchain object
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Manual use of the RAG pipeline
def retrieval_chain_without_LCEL(query: str):
    # Step 1: Retrieve relevant docs
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning?"

    # ======================================================================
    # Option 0: Raw invocation without RAG
    # ======================================================================
    print("\n" + "="*70)
    print("Option 0: Raw invocation without RAG")
    print("="*70)

    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer: ")
    print(result_raw.content)


    # ======================================================================
    # Option 1: Manual use of the RAG pipeline
    # ======================================================================
    print("\n" + "="*70)
    print("Manual use of the RAG pipeline")
    print("="*70)

    result_manual = retrieval_chain_without_LCEL(query)
    print("\nResult:")
    print(result_manual)