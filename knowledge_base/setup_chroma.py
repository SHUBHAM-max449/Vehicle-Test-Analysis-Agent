from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from knowledge_base.failure_reports import failure_reports
from dotenv import load_dotenv

load_dotenv()

def setup_knowledge_base():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_texts(
        texts=failure_reports,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="failure_reports"
    )
    
    print(f"Knowledge base setup complete. {len(failure_reports)} reports embedded.")
    return vectorstore


def load_knowledge_base():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="failure_reports"
    )
    
    return vectorstore


if __name__ == "__main__":
    setup_knowledge_base()