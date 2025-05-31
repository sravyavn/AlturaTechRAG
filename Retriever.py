#using retrieved top k chunks from chosen table and query, ask llm to process output

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from Router import route_query

def retrieve_and_answer(query, index_name, top_k=4):
    embedding_model = SentenceTransformer("BAAI/bge-small-en")
    pc = Pinecone()
    llm = ChatGroq(temperature=0, model="llama3-70b-8192")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant answering internal policy questions. Use ONLY the provided document context to answer accurately.

Context:
{context}

Question:
{question}

Answer:"""
    )
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        
    index = pc.Index(index_name)
    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    context = "\n\n".join([match.metadata["text"] for match in result.matches])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    return response.content.strip(), [match.metadata for match in result.matches]
