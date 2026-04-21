import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    # print(os.environ['INDEX_NAME'])

    # load doc
    loader = TextLoader(
        "C:/python-projects/lchain-agai-folders/sec9-gist-of-RAG/mediumblog1.txt",
        encoding='utf-8')
    loaded_doc = loader.load()
    # print(loaded_doc)

    # split doc into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(loaded_doc)
    # print(f"Created {len(chunks)} chunks")

    """
     embed chunks
     embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
     Use ollama embeddings instead of OpenAI!
     Note that we had to use qwen3 instead of nomic-embed-text because it supports vector dims 32 to 4096 while the latter only supports
    754....
    """
    embeddings = OllamaEmbeddings(model='qwen3-embedding', dimensions=1536)
    
    # Given the chunks of text and the embeddings object, we give these to langchain_pinecone and it will take care of injecting the 
    # text vector pairs into the v db!
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ['INDEX_NAME'])


    # store embeddings in db

