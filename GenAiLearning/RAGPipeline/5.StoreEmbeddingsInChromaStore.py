from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader,PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_pdf_return_documents(file_path="VaibhavZodge.pdf"):
    """Load a PDF file and return its content."""
    docs=[]
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    id=1
    for document in documents:
        docs.append(Document(id=id,page_content=document.page_content))
        id=id+1
        
    return docs

obj_embeddings=OllamaEmbeddings(model="llama3")

vector_store=Chroma(embedding_function=obj_embeddings,collection_name="pdf_embeddings",persist_directory="C:\\gen-ai-learning\\GenAiLearning\\RAGPipeline\\chroma_db")

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

docs=splitter.split_documents(load_pdf_return_documents())

vector_store.add_documents(docs)

similar_vectors=vector_store.search("Does vaibhav have github ?",search_type="similarity",k=5)

print("\n\n******************** Similar vectors :"+str(similar_vectors))   
        