import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

MODEL_EMB_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "students_info.index"
META_FILE = "students_metadata.npy"

class StudentRetriever:
    def __init__(self, index_file: str = INDEX_FILE, meta_file: str = META_FILE, model_name: str = MODEL_EMB_NAME):
        """
        This class loads a FAISS index and corresponding metadata,
        and allows retrieval of the most relevant text chunks for a given student
        using semantic search with sentence embeddings.

        Args:
            index_file (str): Path to the FAISS index file.
            meta_file (str): Path to the metadata file (.npy) containing information about the text chunks.
            model_name (str): Name of the sentence-transformers embedding model.
        """
        self.index = faiss.read_index(index_file)
        self.metadata = np.load(meta_file, allow_pickle=True)
        self.model = SentenceTransformer(model_name)
        self.number_of_chunks_index = self.index.ntotal
        print(f"Number of chunks in index: {self.number_of_chunks_index}")

    def retrieve(self, query: str, student_id: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant text chunks for a specific student based on a query.

        Args:
            query (str): The input query or description used for retrieval.
            student_id (str): Identifier of the student whose chunks should be retrieved.
            top_k (int, optional): Maximum number of chunks to return. Defaults to 5.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieved chunks.
                Each dictionary has the following keys:
                - "score" (float): Similarity score of the chunk with respect to the query.
                - "text" (str): The text content of the chunk.
                - "chunk_id" (str): Unique identifier of the chunk.
        """
        # Generate query embedding
        emb = self.model.encode([query], convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        
        D, I = self.index.search(emb, self.number_of_chunks_index)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: 
                continue
            meta = self.metadata[idx]
            if meta.get("id_student") != student_id:
                continue
            results.append({
                "score": float(score),
                "text": meta.get("text", ""),
                "chunk_id": meta.get("chunk_id")
            })
            if len(results) >= top_k:
                break

        return results

if __name__ == "__main__":
    retriever = StudentRetriever()
    student_id = "student_1"
    query = "The student's skills and knowledge in school subjects such as mathematics, reading, writing, and science. Includes comprehension, application of concepts, problem-solving, and clear expression."
    results = retriever.retrieve(query, student_id=student_id, top_k=3)
