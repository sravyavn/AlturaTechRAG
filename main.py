#take user input, load files , do chunking , store in pinecone in diff tables. pass the input to router.py

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from Router import route_query  
from Retriever import retrieve_and_answer
from Compliance import check_compliance


load_dotenv()

converter = DocumentConverter()

pc = Pinecone()
embedding_model = SentenceTransformer("BAAI/bge-small-en")

tokenizer = AutoTokenizer.from_pretrained("fxmarty/tiny-llama-fast-tokenizer")

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=250, merge_peers=True)


sources = {
    "hr-handbook": converter.convert("HR_Handbook.pdf").document,
    "sales-playbook": converter.convert("Sales_Playbook.pdf").document,
    "security-protocol": converter.convert("Security_Protocol.pdf").document,
    "engineering-sop": converter.convert("https://datasense78.github.io/engineeringsop/").document
}

# create table

# for name in sources.keys():
#     pc.create_index(
#         name=name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# print("✅ Pinecone tables created.")


for name, document in sources.items():
    chunks = list(chunker.chunk(document)) 
    index = pc.Index(name)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk.text, normalize_embeddings=True)

        headings = chunk.meta.headings[0] if chunk.meta.headings else ""
        page_no = (
            chunk.meta.doc_items[0].prov[0].page_no
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov and chunk.meta.doc_items[0].prov[0].page_no is not None
            else 0
        )
        filename = chunk.meta.origin.filename if chunk.meta.origin and chunk.meta.origin.filename else ""

        index.upsert([
            (
                str(i + 1),
                embedding.tolist(),
                {
                    "text": chunk.text,
                    "heading": headings,
                    "page_no": page_no,
                    "source": filename
                }
            )
        ])

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    best_source = route_query(user_question)
    print(best_source)
    
    retrieved_answer, context_metadata = retrieve_and_answer(user_question, best_source)

    final_answer = check_compliance(retrieved_answer)

    print("\nFinal Answer:\n")
    print(final_answer)
