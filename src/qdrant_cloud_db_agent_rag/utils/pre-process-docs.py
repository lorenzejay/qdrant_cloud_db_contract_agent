import os
import glob
import uuid

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv
from qdrant_client.models import PointStruct

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


def generate_chunks_and_metadata(file_path):
    document_converter = DocumentConverter()
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    )
    source = Path(file_path)
    converted_documented = document_converter.convert(source).document
    chunk_iterable = chunker.chunk(converted_documented)

    chunk_info = extract_chunk_info(chunk_iterable)
    for index, chunk in enumerate(chunk_info):
        print(f"Chunk {index + 1}:")
        print(f"Text: {chunk['text']}")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 50)

    return chunk_info


def extract_chunk_info(chunks):
    results = []
    for chunk in chunks:
        if isinstance(chunk.text, str):
            # Create entry for each chunk of text
            entry = {
                "text": {"content": chunk.text},
                "metadata": {"page_number": None, "filename": None, "headings": None},
            }

            # Extract metadata from the chunk
            if chunk.meta:
                # Get page number from first provenance item if available
                if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                    entry["metadata"]["page_number"] = (
                        chunk.meta.doc_items[0].prov[0].page_no
                    )

                # Get filename from origin if available
                if chunk.meta.origin and chunk.meta.origin.filename:
                    entry["metadata"]["filename"] = chunk.meta.origin.filename

                # Get headings if available
                if chunk.meta.headings:
                    entry["metadata"]["headings"] = chunk.meta.headings

            results.append(entry)

    return results


if __name__ == "__main__":
    files = glob.glob("knowledge/contracts/*.pdf", recursive=True)
    # print("files", files)

    # Create client and collection once, outside the file loop
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    # print("client", client, qdrant_url, qdrant_api_key)

    # Create collection once
    # try:
    #     client.create_collection(
    #         collection_name="contracts",
    #         vectors_config=models.VectorParams(
    #             size=100, distance=models.Distance.COSINE
    #         ),
    #     )
    # except Exception as e:
    #     print(f"Collection creation error (might already exist): {e}")

    # Process each file
    for file in files:
        data = generate_chunks_and_metadata(file)
        print("data", data)

        embedding_model = TextEmbedding()
        try:
            for d in data:
                print("d1:", d)
                document = d["text"]["content"]
                print("document content:", document)
                metadata = d["metadata"]
                # Convert generator to list and get first embedding
                # document_embedding = list(embedding_model.embed([document]))[0]
                # print("document_embedding", document_embedding)

                # Generate a unique ID for each document
                doc_id = str(uuid.uuid4())
                # point = PointStruct(
                #     id=doc_id,
                #     vector=document_embedding,
                #     payload=metadata,
                # )
                client.add(
                    collection_name="contracts",
                    documents=[document],
                    metadata=metadata,
                    ids=[doc_id],
                )
            print("added to qdrant successfully")
        except Exception as e:
            print("error", e)
