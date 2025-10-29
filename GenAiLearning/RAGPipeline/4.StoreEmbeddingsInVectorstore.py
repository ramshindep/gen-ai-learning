from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


def load_pdf_return_documents(file_path="VaibhavZodge.pdf"):
    """Load a PDF file and return its content. """
    docs=[]
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    id=1
    for document in documents:
        docs.append(Document(id=id,page_content=document.page_content))
        id=id+1
    return docs


obj_embeddings=OllamaEmbeddings(model="llama3")

vector_store= InMemoryVectorStore(embedding=obj_embeddings)

vector_store.add_documents(load_pdf_return_documents())

similar_vectors=vector_store.search("Does vaibhav have github ? ",search_type="similarity",k=1)

print("\n\n****************** Similar Vectors :"+str(similar_vectors))