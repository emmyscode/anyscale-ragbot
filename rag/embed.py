import os

from langchain_openai import OpenAIEmbeddings


class EmbedChunks:
    def __init__(self, model_name):
        # Embedding model
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}
