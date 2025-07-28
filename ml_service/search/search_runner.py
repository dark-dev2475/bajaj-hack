# search/search_runner.py

import logging
from typing import List, Dict, Any, Optional
import asyncio

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_pinecone import PineconeVectorStore

from clients import pinecone_client, openai_async_client
from query_parser.schema import PolicyQuery
# Assuming a translator function exists from your previous code
from .translator import translate_to_english_async

# --- 1. Define Metadata Fields for the Retriever ---
# This tells the retriever what fields it can filter on in your vector database.
metadata_field_info = [
    AttributeInfo(
        name="policy_type",
        description="The type of insurance policy, such as 'personal accident', 'health', or 'travel'.",
        type="string",
    ),
    AttributeInfo(
        name="location",
        description="The city where the event occurred.",
        type="string",
    ),
    # Add other filterable fields from your metadata here if needed
]

# --- 2. Initialize Core Components ---
# These are the building blocks for our retriever.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


async def perform_search_async(
    raw_query: str,
    index_name: str,
    namespace: str,
    top_k: int = 5, # It's often better to retrieve more docs (e.g., 5) for the LLM to get more context
) -> List[Dict[str, Any]]:
    """
    Performs an advanced search using LangChain's Self-Querying Retriever.
    It first translates the query to English to ensure reliable filter generation.
    """
    logging.info(f"Performing self-query search for: '{raw_query}' in namespace '{namespace}'")
    
    try:
        # --- ADDED: Translate query to English for reliability ---
        english_query = await translate_to_english_async(raw_query, openai_async_client)
        logging.info(f"Translated query to English: '{english_query}'")
        # --- END OF ADDITION ---

        # --- 3. Set up the Pinecone Vector Store as a LangChain object ---
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_client.api_key, # Assumes your client has the key
            namespace=namespace
        )

        # --- 4. Create the Self-Querying Retriever ---
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents="The content of an insurance policy clause.",
            metadata_field_info=metadata_field_info,
            verbose=True, # Set to True for debugging to see the generated queries
            k=top_k
        )

        # --- 5. Invoke the Retriever ---
        # We run this in a thread because the underlying LangChain invocation can have sync parts.
        retrieved_docs = await asyncio.to_thread(
            retriever.invoke,
            english_query # Use the translated query
        )

        logging.info(f"Retrieved {len(retrieved_docs)} documents using self-query.")
        
        # --- 6. Format the results ---
        # Convert the LangChain Document objects back to the dictionary format your pipeline expects.
        results = [
            {
                "id": doc.metadata.get("id", ""),
                "score": doc.metadata.get("_score", 0), # LangChain might not always provide a score
                "metadata": doc.metadata,
                "chunk_text": doc.page_content
            }
            for doc in retrieved_docs
        ]
        return results

    except Exception as e:
        logging.exception(f"An error occurred during self-query retrieval: {e}")
        return [{"error": "Failed to perform search."}]
