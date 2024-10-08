import os
from datetime import datetime
from functools import partial
from pathlib import Path

import psycopg
import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pgvector.psycopg import register_vector

from rag.config import EFS_DIR
from rag.data import extract_md_sections, extract_sections
from rag.embed import EmbedChunks
from rag.utils import execute_bash


class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(
            "dbname=postgres user=postgres host=localhost password=postgres"
        ) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(
                    batch["text"], batch["source"], batch["embeddings"]
                ):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)",
                        (
                            text,
                            source,
                            embedding,
                        ),
                    )
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def chunk_md(md_doc):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    chunks = markdown_splitter.split_text(md_doc["text"])
    return [{"text": chunk.page_content, "source": md_doc["source"]} for chunk in chunks]


def build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, sql_dump_fp):
    # Check if it's Anyscale or Ray
    if "docs.ray.io" in str(docs_dir):
        # docs -> sections -> chunks
        ds = ray.data.from_items(
            [{"path": path} for path in docs_dir.rglob("*.html") if not path.is_dir()]
        )
        sections_ds = ds.flat_map(extract_sections)
        chunks_ds = sections_ds.flat_map(
            partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    elif "docs.anyscale.com" in str(docs_dir):
        # md files -> sections -> md chunks
        sections_ds = ray.data.from_items(extract_md_sections(docs_dir))
        chunks_ds = sections_ds.flat_map(chunk_md)
    else:
        raise ValueError(f"Unrecognized documentation directory: {docs_dir}")

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        batch_size=100,
        num_gpus=1,
        concurrency=1,
    )

    # Index data
    embedded_chunks.map_batches(
        StoreResults,
        batch_size=128,
        num_cpus=1,
        concurrency=6,
    ).count()

    # Save to SQL dump
    execute_bash(f"sudo -u postgres pg_dump -c > {sql_dump_fp}")
    print(f"Updated the index for {docs_dir.name}!")


def load_index(embedding_model_name, embedding_dim, chunk_size, chunk_overlap, sql_dump_fp=None):
    # Drop current Vector DB and prepare for new one
    execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -c "DROP TABLE document;"')
    execute_bash(f"sudo -u postgres psql -f ../migrations/vector-{embedding_dim}.sql")

    if not sql_dump_fp:
        date_str = datetime.now().strftime("%Y%m%d")
        sql_dump_fp = Path(EFS_DIR, "sql_dumps", f"{date_str}.sql")

    # Vector DB
    if sql_dump_fp.exists():  # Load from SQL dump
        execute_bash(f'psql "{os.environ["DB_CONNECTION_STRING"]}" -f {sql_dump_fp}')
    else:  # Create new index
        # Iterate over both docs directories in the root_dir
        for docs_dir in Path(EFS_DIR).iterdir():
            build_index(
                docs_dir=docs_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model_name=embedding_model_name,
                sql_dump_fp=sql_dump_fp,
            )

    # Chunks
    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id, text, source FROM document")
            chunks = cur.fetchall()
    return chunks
