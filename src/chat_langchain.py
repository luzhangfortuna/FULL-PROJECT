from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone
from .models import db, ChatMessage
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import logging
import os

# Load environment variables
load_dotenv('keys.env')

# Access environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Debugging: Print the API keys
print(f"Pinecone API Key: {PINECONE_API_KEY}")
print(f"OpenAI API Key: {OPENAI_API_KEY}")

# Check if API keys are set
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is not set. Please check your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set. Please check your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize index
index_name = 'mergers-and-acquisitions-in-tech'
if index_name not in pc.list_indexes().names():
    logger.info(f"Index '{index_name}' does not exist. Creating it now...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimension for OpenAI embeddings
        metric='cosine',
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
else:
    logger.info(f"Index '{index_name}' already exists.")

# Initialize the Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load and process PDF documents
try:
    pdf_loader = DirectoryLoader('./src/pdf', glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    logger.info(f"Loaded {len(pdf_documents)} PDF documents.")
except Exception as e:
    logger.error(f"Error loading PDF documents: {e}")
    pdf_documents = []

# Split PDF documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pdf_chunks = text_splitter.split_documents(pdf_documents)

# Load and process CSV data
try:
    csv_path = "./src/Tech M&A Deals (1988-2021).csv"
    data = pd.read_csv(csv_path)

    # Replace NaN values with empty strings
    data = data.fillna("")
    logger.info(f"Loaded CSV data with {len(data)} rows.")
except Exception as e:
    logger.error(f"Error loading CSV data: {e}")
    data = pd.DataFrame()

# Convert CSV rows into Document objects
csv_documents = []
for _, row in data.iterrows():
    try:
        # Create a meaningful page_content for each row
        page_content = (
            f"Parent Company: {row['Parent Company']}\n"
            f"Acquired Company: {row['Acquired Company']}\n"
            f"Business: {row['Business']}\n"
            f"Country: {row['Country']}\n"
            f"Acquisition Price: {row['Acquisition Price']}\n"
            f"Category: {row['Category']}\n"
            f"Derived Products: {row['Derived Products']}"
        )

        # Create metadata for each row
        metadata = {
            "parent_company": row["Parent Company"],
            "acquired_company": row["Acquired Company"],
            "acquisition_year": row["Acquisition Year"],
            "acquisition_month": row["Acquisition Month"],
            "business": row["Business"],
            "country": row["Country"],
            "acquisition_price": row["Acquisition Price"],
            "category": row["Category"],
            "derived_products": row["Derived Products"]
        }

        # Create a Document object
        doc = Document(page_content=page_content, metadata=metadata)
        csv_documents.append(doc)
    except Exception as e:
        logger.error(f"Error processing CSV row: {e}")

# Split CSV documents into chunks (if needed)
csv_chunks = text_splitter.split_documents(csv_documents)

# Combine PDF and CSV chunks
all_chunks = pdf_chunks + csv_chunks

# Generate embeddings and upload to Pinecone
try:
    logger.info("Creating embeddings and uploading to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    logger.info("Embeddings uploaded to Pinecone successfully!")
except Exception as e:
    logger.error(f"Error uploading embeddings to Pinecone: {e}")

# construct 
template = """" Today is {today}.  
You are Luna, a virtual market researcher specializing in the tech industry, created by LU.  
Your expertise lies in analyzing market trends, mergers and acquisitions (M&A) activities, and providing insights into strategic opportunities within the tech sector.  

You deliver responses that are professional, concise, and backed by relevant data. Your tone reflects a deep understanding of industry dynamics, emphasizing clarity and practicality. Answer each question truthfully to the best of your abilities based on the provided information, focusing on:  

- Market trends and growth opportunities in the tech sector.  
- M&A activities, including key players, motivations, and outcomes.  
- Strategic insights derived from market data and case studies.  
- Competitive landscape analysis within the tech industry.  
- Potential risks and opportunities linked to M&A transactions.  

Structure your responses as follows:  
1. Brief summary of the key point(s).  
2. Detailed insights presented in bullet points, highlighting actionable recommendations or implications where applicable.  
3. Where relevant, include examples or data points to contextualize your analysis.  

Your goal is to support businesses, investors, and stakeholders in making informed decisions in the tech industry's evolving market.  

<context>  
{context}  
</context>  

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Create conversation chain
llm = ChatOpenAI(streaming=True)
memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
        return_source_documents=True,
        output_key="answer"
    )
    

# Chat function with streaming and DB storage
def call_chat(question):
    result = conversation.invoke({"question": question})
    
    answer = result["answer"]
    sources = result.get("source_documents", [])
    
    # Add source information to the answer
    source_info = "\n\nSources:"
    for doc in sources:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'unknown')
        #source_info += f"\n- {source}, page {page}"
    
    #full_response = answer + source_info
    full_response = answer
    
    # Stream the response
    for chunk in full_response.split():
        yield {"token": chunk + " "}
    
    # Save to database with source information
    chat_message = ChatMessage(
        user_id=1,
        question=question,
        answer=full_response
    )
    db.session.add(chat_message)
    db.session.commit()