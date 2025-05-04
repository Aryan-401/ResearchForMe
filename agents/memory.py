from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import JinaEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Dict, Tuple
import uuid


class Memory:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.memory = InMemoryVectorStore(embedding=self.embeddings)
        # self.user_id = "Aryan"
        self.text_splitter = SemanticChunker(self.embeddings)

    def save_recall_memory(self, memory: str, metadata: Dict[str, str]) -> str:
        """Save memory to vectorstore for later semantic retrieval."""
        document = Document(
            page_content=memory, id=str(uuid.uuid4()), metadata=metadata
        )
        self.memory.add_documents([document])
        return memory

    def search_recall_memories(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant memories."""

        documents = self.memory.similarity_search(
            query, k=k
        )
        return documents

    def save_chunked_memory(self, memory: str, metadata: Dict[str, str]):
        lod = self.text_splitter.create_documents(
            [
                memory
            ],
            metadatas=[metadata]
        )
        print("Documents from Webpage:", len(lod))
        self.memory.add_documents(lod)
        return memory

    def get_sim_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Get the similarity score of the memories."""
        documents = self.memory.similarity_search_with_score(query, k=k)
        return documents

    def delete_id(self, id: str):
        """Clear the vectorstore."""
        self.memory.delete(
            ids=[id]
        )
