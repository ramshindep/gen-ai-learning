from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader
from langchain_core.documents import Document

def load_pdf(file_path="./VaibhavZodge.pdf"):
   """Load a PDF file and return its content."""
   loader=PyPDFLoader(file_path)
   Documents=loader.load()
   return Documents


def load_pdf_return_documents(file_path="./VaibhavZodge.pdf"):
    """Load a PDF file and return its content."""
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    docs=[]
    id=1
    for document in documents:
        docs.append(Document(id=id,page_content=document.page_content))
        id=id+1
    return docs
 
def load_from_web(url=""):
    loader=WebBaseLoader(url)
    Documents=loader.load()
    return Documents

def load_pdf_directory(directory_path="./VaibhavZodge.pdf"):
    """Load all pdf files from a directory and return their content."""
    loader=PyPDFDirectoryLoader(directory_path)
    documents=loader.load()
    return documents

documents=load_from_web()
print("\n\n *************** web: "+str(documents))