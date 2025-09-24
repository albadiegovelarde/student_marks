import os
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class StudentIndexBuilder:
    def __init__(
        self,
        yaml_dir: str = "data/student_reports",
        index_file: str = "student_marks/students_info.index",
        meta_file: str = "student_marks/students_metadata.npy",
        model_emb_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 150,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the StudentIndexBuilder class.
        """
        self.yaml_dir = yaml_dir
        self.index_file = index_file
        self.meta_file = meta_file
        self.model_emb_name = model_emb_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def load_yaml_files(self) -> List[Dict]:
        """
        Load student reports from YAML files in the specified directory.

        Returns:
            List[Dict]: A list of dictionaries containing the data from each YAML file.
        """
        docs = []
        for file in os.listdir(self.yaml_dir):
            if file.endswith(".yaml") or file.endswith(".yml"):
                path = os.path.join(self.yaml_dir, file)
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    docs.append(data)
        return docs

    def chunk_text(self, text: str) -> List[str]:
        """
        Split a text into chunks of fixed length with optional overlap.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize a set of vectors using L2 normalization.

        Args:
            vectors (np.ndarray): Array of embedding vectors.

        Returns:
            np.ndarray: Array of normalized vectors.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)

    def build_index(self):
        """
        Build a FAISS index from student reports.
        """
        print("Uploading student YAML files")
        docs = self.load_yaml_files()

        print("Generating chunks")
        all_chunks = []
        for doc in docs:
            report = doc.get("report", "")
            student_id = doc.get("id", "no_id")

            chunks = self.chunk_text(report)
            for i, ch in enumerate(chunks):
                all_chunks.append({
                    "id_student": student_id,
                    "chunk_id": f"{student_id}_chunk_{i:04d}",
                    "text": ch,
                })

        print(f"Total chunks: {len(all_chunks)}")

        # Obtain embeddings
        print("Generating embeddings")
        model_emb = SentenceTransformer(self.model_emb_name)
        texts = [c["text"] for c in all_chunks]
        embeddings = model_emb.encode(texts, convert_to_numpy=True)
        embeddings = self.normalize(embeddings)

        # Create index FAISS
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        # Save index and metadata
        faiss.write_index(index, self.index_file)
        np.save(self.meta_file, all_chunks)


if __name__ == "__main__":
    builder = StudentIndexBuilder()
    builder.build_index()
