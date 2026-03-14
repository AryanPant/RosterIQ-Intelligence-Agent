import json
import os
import re

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SemanticMemory:

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_DIM = 384

    def __init__(self, source_path=None, metadata_path=None, index_path=None):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_index_path = os.path.join(base_dir, "memory", "semantic_memory.index")
        default_metadata_path = os.path.join(base_dir, "memory", "semantic_memory_store.json")
        self.index_path = index_path or default_index_path
        self.metadata_path = metadata_path or default_metadata_path
        self.source_json_path = self._resolve_source_json(base_dir, source_path=source_path)
        self.schema_entries = self._load_semantic_entries()
        self.embedding_model = self._load_embedding_model()
        self.embedding_dim = self._determine_embedding_dim()
        self.index = self._load_or_create_index()
        self._sync_index()

    def _resolve_source_json(self, base_dir, source_path=None):

        candidate_paths = [
            source_path,
            os.getenv("ROSTERIQ_SEMANTIC_MEMORY_SOURCE"),
            os.path.join(base_dir, "memory", "semantic_memory.json"),
            r"c:\Users\ASUS\Desktop\semantic_memory.json",
        ]
        for path in candidate_paths:
            if path and os.path.exists(path):
                return path
        return None

    @staticmethod
    def _normalize_store_entry(entry):

        if not isinstance(entry, dict):
            return None

        category = str(entry.get("category") or "note").strip() or "note"
        name = str(entry.get("name") or entry.get("topic") or "unknown").strip() or "unknown"
        description = str(entry.get("description") or entry.get("body") or "").strip()
        chunk = str(entry.get("chunk") or "").strip()
        if not chunk:
            label = category.upper()
            chunk = f"{label}: {name}\nDESCRIPTION: {description}".strip()

        return {
            "category": category,
            "name": name,
            "chunk": chunk,
            "description": description,
        }

    def _load_entries_from_store(self):

        if not os.path.exists(self.metadata_path):
            return []

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(payload, list):
            return []

        entries = []
        for item in payload:
            normalized = self._normalize_store_entry(item)
            if normalized is not None:
                entries.append(normalized)
        return entries

    def _load_semantic_entries(self):

        if not self.source_json_path:
            return self._load_entries_from_store()

        try:
            with open(self.source_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return self._load_entries_from_store()

        if isinstance(payload, list):
            entries = []
            for item in payload:
                normalized = self._normalize_store_entry(item)
                if normalized is not None:
                    entries.append(normalized)
            return entries

        entries = []
        for key, description in payload.get("terms", {}).items():
            entries.append(
                {
                    "category": "term",
                    "name": key,
                    "chunk": f"TERM: {key}\nDESCRIPTION: {description}",
                    "description": description,
                }
            )

        for key, description in payload.get("stages", {}).items():
            entries.append(
                {
                    "category": "stage",
                    "name": key,
                    "chunk": f"STAGE: {key}\nDESCRIPTION: {description}",
                    "description": description,
                }
            )

        for note in payload.get("notes", []):
            topic = note.get("topic", "unknown_topic")
            body = note.get("body", "")
            entries.append(
                {
                    "category": "note",
                    "name": topic,
                    "chunk": f"NOTE: {topic}\nDESCRIPTION: {body}",
                    "description": body,
                }
            )

        return entries

    def _load_embedding_model(self):

        if SentenceTransformer is None:
            return None

        try:
            return SentenceTransformer(self.EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            return None

    def _determine_embedding_dim(self):

        if self.embedding_model is None:
            return self.DEFAULT_EMBEDDING_DIM

        try:
            sample = self.embedding_model.encode(["sample"], normalize_embeddings=True)
            return int(sample.shape[1])
        except Exception:
            return self.DEFAULT_EMBEDDING_DIM

    def _load_or_create_index(self):

        if faiss is None:
            return None

        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception:
                pass

        return faiss.IndexFlatIP(self.embedding_dim)

    def _save_metadata(self):

        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(self.schema_entries, handle, indent=2)

    def _save_index(self):

        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def _entry_text(self, entry):

        return entry["chunk"]

    def _sync_index(self):

        self._save_metadata()
        if self.index is None or self.embedding_model is None:
            return

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        vectors = []
        for entry in self.schema_entries:
            vector = self.embed_text(self._entry_text(entry))
            if vector is not None:
                vectors.append(vector[0])

        if vectors:
            self.index.add(np.asarray(vectors, dtype="float32"))
            self._save_index()

    def _tokenize(self, text):

        return set(re.findall(r"[A-Za-z0-9_]+", (text or "").lower()))

    def embed_text(self, text):

        if self.embedding_model is None:
            return None

        try:
            vector = self.embedding_model.encode([text], normalize_embeddings=True)
            return np.asarray(vector, dtype="float32")
        except Exception:
            return None

    def explain(self, key):

        for entry in self.schema_entries:
            if entry["name"] == key:
                return entry["description"]
        return ""

    def query_hybrid(self, query, alpha=0.5, limit=15):

        lexical_scores = []
        query_tokens = self._tokenize(query)
        for index, entry in enumerate(self.schema_entries):
            entry_tokens = self._tokenize(self._entry_text(entry))
            lexical_overlap = len(query_tokens.intersection(entry_tokens))
            lexical_scores.append((index, float(lexical_overlap)))

        semantic_scores = {}
        query_vector = self.embed_text(query)
        if self.index is not None and query_vector is not None and self.index.ntotal:
            search_limit = min(limit * 3, len(self.schema_entries))
            distances, positions = self.index.search(query_vector, search_limit)
            for distance, position in zip(distances[0], positions[0]):
                if 0 <= position < len(self.schema_entries):
                    semantic_scores[int(position)] = float(distance)

        combined = []
        for index, lexical_score in lexical_scores:
            semantic_score = semantic_scores.get(index, 0.0)
            hybrid_score = (alpha * semantic_score) + ((1 - alpha) * lexical_score)
            combined.append((hybrid_score, self.schema_entries[index]))

        combined.sort(key=lambda item: item[0], reverse=True)
        return [entry for score, entry in combined[:limit] if score > 0]

    def semantic_recall(self, query, alpha=0.5, limit=15):

        memories = self.query_hybrid(query=query, alpha=alpha, limit=limit)
        combined_text = ""

        for index, memory in enumerate(memories):
            combined_text += f"\nCHUNK {index + 1}:\n"
            combined_text += self._entry_text(memory).strip()

        return combined_text.strip()
