import json
import os
import sqlite3
from datetime import datetime

import numpy as np
from utils.openrouter_client import OpenRouterClient

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EpisodicMemory:

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_DIM = 384
    MAX_ENTRIES = 100

    def __init__(self, storage_path=None, index_path=None):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_storage_path = os.getenv("ROSTERIQ_EPISODIC_MEMORY_PATH")
        configured_path = storage_path or env_storage_path or os.path.join(
            base_dir,
            "memory",
            "episodic_memory_store.db",
        )
        self.storage_path = self._resolve_storage_path(configured_path)
        self.index_path = index_path or os.path.splitext(self.storage_path)[0] + ".index"

        self.embedding_model = self._load_embedding_model()
        self.embedding_dim = self._determine_embedding_dim()
        self.index = self._load_or_create_index()
        self.llm = OpenRouterClient()

        self._initialize_database()
        self.entries = self._load_entries_from_db()
        self._sync_index_with_entries()

    def _resolve_storage_path(self, configured_path):

        if configured_path.lower().endswith(".json"):
            return os.path.splitext(configured_path)[0] + ".db"
        return configured_path

    def _connect(self):

        connection = sqlite3.connect(self.storage_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self):

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    context_tags_json TEXT NOT NULL,
                    conversation_summary TEXT NOT NULL,
                    what_worked TEXT NOT NULL,
                    what_to_avoid TEXT NOT NULL,
                    embedding BLOB
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_memory_timestamp
                ON episodic_memory(timestamp DESC)
                """
            )
            connection.commit()

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
            sample_vector = self.embedding_model.encode(["sample"], normalize_embeddings=True)
            return int(sample_vector.shape[1])
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

    def _save_index(self):

        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def _is_meta_memory_entry(self, entry):

        metadata = entry.get("metadata", {})
        intents = set(metadata.get("intents", []))
        response = (entry.get("response") or "").strip().lower()

        if "memory_lookup" in intents:
            return True

        if response.startswith("yes. we previously investigated") or response.startswith(
            "no similar prior investigation was found"
        ):
            return True

        return False

    def _load_entries_from_db(self):

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT timestamp, query, response, metadata_json, context_tags_json,
                       conversation_summary, what_worked, what_to_avoid
                FROM episodic_memory
                ORDER BY timestamp ASC, id ASC
                """
            ).fetchall()

        entries = []
        for row in rows:
            entry = {
                "timestamp": row["timestamp"],
                "query": row["query"],
                "response": row["response"],
                "metadata": self._parse_json_field(row["metadata_json"], default={}),
                "context_tags": self._parse_json_field(row["context_tags_json"], default=[]),
                "conversation_summary": row["conversation_summary"],
                "what_worked": row["what_worked"],
                "what_to_avoid": row["what_to_avoid"],
            }
            if not self._is_meta_memory_entry(entry):
                entries.append(entry)
        return entries[-self.MAX_ENTRIES :]

    @staticmethod
    def _parse_json_field(value, default):

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(default)) else default
        except Exception:
            return default

    @staticmethod
    def _serialize_embedding(vector):

        if vector is None:
            return None
        return np.asarray(vector[0], dtype="float32").tobytes()

    def _load_embedding_from_row(self, row):

        embedding_blob = row["embedding"]
        if embedding_blob is None:
            return None
        try:
            vector = np.frombuffer(embedding_blob, dtype="float32")
            if vector.size != self.embedding_dim:
                return None
            return vector
        except Exception:
            return None

    def _stringify_field(self, value):

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return " ".join(
                part for part in (self._stringify_field(item).strip() for item in value) if part
            )
        if isinstance(value, dict):
            try:
                return json.dumps(value, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)

    def _response_text_for_embedding(self, entry):

        metadata = entry.get("metadata", {})
        semantic_tags = (
            self._stringify_field(metadata.get("topics", []))
            + " "
            + self._stringify_field(metadata.get("intents", []))
        )
        return " ".join(
            [
                self._stringify_field(entry.get("response", "")),
                self._stringify_field(entry.get("query", "")),
                self._stringify_field(entry.get("conversation_summary", "")),
                self._stringify_field(entry.get("context_tags", [])),
                self._stringify_field(entry.get("what_worked", "")),
                self._stringify_field(entry.get("what_to_avoid", "")),
                self._stringify_field(metadata.get("market", "")),
                semantic_tags.strip(),
            ]
        ).strip()

    def _fallback_reflection(self, query, result, metadata=None):

        metadata = metadata or {}
        tags = []
        if metadata.get("market"):
            tags.append(str(metadata["market"]).lower())
        tags.extend(metadata.get("topics", [])[:2])
        tags.extend(metadata.get("intents", [])[:2])
        tags = [tag for tag in tags if tag]

        return {
            "context_tags": list(dict.fromkeys(tags))[:4] or ["roster_operations", "diagnostics"],
            "conversation_summary": (result or query or "").strip()[:220],
            "what_worked": "Ground the answer in pipeline evidence, market metrics, and stored investigation context.",
            "what_to_avoid": "Do not answer memory questions with recursive memory summaries or ignore market-specific filtering.",
        }

    def _reflect_episode(self, query, result, metadata=None):

        metadata = metadata or {}
        prompt = f"""
You are creating an episodic memory reflection for a healthcare roster intelligence agent.
Given the user query, final answer, and metadata, produce JSON with keys:
context_tags, conversation_summary, what_worked, what_to_avoid.

Query: {query}
Answer: {result}
Metadata: {json.dumps(metadata)}
""".strip()

        raw = self.llm.generate(prompt)
        if not raw:
            return self._fallback_reflection(query, result, metadata)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            return {
                "context_tags": [
                    self._stringify_field(tag) for tag in parsed.get("context_tags", [])[:4]
                ],
                "conversation_summary": self._stringify_field(parsed.get("conversation_summary", "")),
                "what_worked": self._stringify_field(parsed.get("what_worked", "")),
                "what_to_avoid": self._stringify_field(parsed.get("what_to_avoid", "")),
            }
        except Exception:
            return self._fallback_reflection(query, result, metadata)

    def format_for_prompt(self, entries, limit=3):

        chunks = []
        for index, entry in enumerate(entries[:limit]):
            chunks.append(
                "\n".join(
                    [
                        f"EPISODE {index + 1}:",
                        f"Query: {self._stringify_field(entry.get('query', ''))}",
                        f"Summary: {self._stringify_field(entry.get('conversation_summary', entry.get('response', '')))}",
                        f"What worked: {self._stringify_field(entry.get('what_worked', ''))}",
                        f"What to avoid: {self._stringify_field(entry.get('what_to_avoid', ''))}",
                        f"Tags: {self._stringify_field(entry.get('context_tags', []))}",
                    ]
                )
            )
        return "\n\n".join(chunks).strip()

    def embed_text(self, text):

        if self.embedding_model is None:
            return None

        try:
            vector = self.embedding_model.encode([text], normalize_embeddings=True)
            return np.asarray(vector, dtype="float32")
        except Exception:
            return None

    def embed_query(self, query):

        return self.embed_text(query)

    def _rebuild_index(self):

        if faiss is None:
            self.index = None
            return

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        vectors = []
        rebuilt_entries = []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, query, response, metadata_json, context_tags_json,
                       conversation_summary, what_worked, what_to_avoid, embedding
                FROM episodic_memory
                ORDER BY timestamp ASC, id ASC
                """
            ).fetchall()

            for row in rows:
                entry = {
                    "timestamp": row["timestamp"],
                    "query": row["query"],
                    "response": row["response"],
                    "metadata": self._parse_json_field(row["metadata_json"], default={}),
                    "context_tags": self._parse_json_field(row["context_tags_json"], default=[]),
                    "conversation_summary": row["conversation_summary"],
                    "what_worked": row["what_worked"],
                    "what_to_avoid": row["what_to_avoid"],
                }
                if self._is_meta_memory_entry(entry):
                    continue

                vector = self._load_embedding_from_row(row)
                if vector is None:
                    regenerated = self.embed_text(self._response_text_for_embedding(entry))
                    vector = np.asarray(regenerated[0], dtype="float32") if regenerated is not None else None
                    if vector is not None:
                        connection.execute(
                            "UPDATE episodic_memory SET embedding = ? WHERE id = ?",
                            (vector.tobytes(), row["id"]),
                        )

                if vector is None:
                    continue

                vectors.append(vector)
                rebuilt_entries.append(entry)

            connection.commit()

        self.entries = rebuilt_entries[-self.MAX_ENTRIES :]
        if vectors:
            matrix = np.asarray(vectors[-self.MAX_ENTRIES :], dtype="float32")
            self.index.add(matrix)
        self._save_index()

    def _sync_index_with_entries(self):

        if faiss is None or self.embedding_model is None:
            return

        current_size = self.index.ntotal if self.index is not None else 0
        if current_size != len(self.entries):
            self._rebuild_index()

    def _semantic_score(self, query_profile, entry):

        if not query_profile:
            return 0

        metadata = entry.get("metadata", {})
        score = 0

        query_market = query_profile.get("market")
        entry_market = metadata.get("market")
        if query_market and entry_market and str(query_market).upper() == str(entry_market).upper():
            score += 0.15

        query_intents = set(query_profile.get("intents", []))
        entry_intents = set(metadata.get("intents", []))
        score += len(query_intents.intersection(entry_intents)) * 0.08

        query_topics = set(query_profile.get("topics", []))
        entry_topics = set(metadata.get("topics", []))
        score += len(query_topics.intersection(entry_topics)) * 0.06

        return score

    def _matches_memory_filters(self, query_profile, entry):

        if not query_profile:
            return True

        query_market = query_profile.get("market")
        entry_market = entry.get("metadata", {}).get("market")

        if query_profile.get("is_memory_query") and query_market:
            if not entry_market or str(query_market).upper() != str(entry_market).upper():
                return False

        return True

    def _fallback_retrieve(self, query, query_profile=None, limit=3):

        normalized_query = set((query or "").lower().split())
        ranked = []
        for entry in self.entries:
            if not self._matches_memory_filters(query_profile, entry):
                continue
            text = self._response_text_for_embedding(entry).lower()
            overlap = len(normalized_query.intersection(set(text.split())))
            score = overlap + self._semantic_score(query_profile, entry)
            if score > 0:
                ranked.append((score, entry))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in ranked[:limit]]

    def _trim_entries(self):

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM episodic_memory
                WHERE id NOT IN (
                    SELECT id
                    FROM episodic_memory
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                )
                """,
                (self.MAX_ENTRIES,),
            )
            connection.commit()

    def store(self, query, result, metadata=None):

        reflection = self._reflect_episode(query, result, metadata)
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "query": query,
            "response": result,
            "metadata": metadata or {},
            "context_tags": reflection.get("context_tags", []),
            "conversation_summary": reflection.get("conversation_summary", ""),
            "what_worked": reflection.get("what_worked", ""),
            "what_to_avoid": reflection.get("what_to_avoid", ""),
        }
        if self._is_meta_memory_entry(entry):
            return

        embedding_vector = self.embed_text(self._response_text_for_embedding(entry))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodic_memory (
                    timestamp,
                    query,
                    response,
                    metadata_json,
                    context_tags_json,
                    conversation_summary,
                    what_worked,
                    what_to_avoid,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["timestamp"],
                    self._stringify_field(entry["query"]),
                    self._stringify_field(entry["response"]),
                    json.dumps(entry["metadata"], sort_keys=True),
                    json.dumps(entry["context_tags"]),
                    self._stringify_field(entry["conversation_summary"]),
                    self._stringify_field(entry["what_worked"]),
                    self._stringify_field(entry["what_to_avoid"]),
                    self._serialize_embedding(embedding_vector),
                ),
            )
            connection.commit()

        self._trim_entries()
        self.entries = self._load_entries_from_db()

        if self.index is None or self.embedding_model is None:
            return

        if self.index.ntotal != len(self.entries) - 1:
            self._rebuild_index()
            return

        if embedding_vector is None:
            self._rebuild_index()
            return

        self.index.add(np.asarray(embedding_vector, dtype="float32"))
        self._save_index()

    def retrieve(self, query, query_profile=None, limit=3):

        if not self.entries:
            return []

        query_vector = self.embed_text(query)
        if self.index is None or query_vector is None or self.index.ntotal == 0:
            return self._fallback_retrieve(query, query_profile=query_profile, limit=limit)

        search_count = min(max(limit * 3, limit), self.index.ntotal)
        similarities, positions = self.index.search(query_vector, search_count)

        ranked = []
        for similarity, position in zip(similarities[0], positions[0]):
            if position < 0 or position >= len(self.entries):
                continue
            entry = self.entries[position]
            if not self._matches_memory_filters(query_profile, entry):
                continue
            score = float(similarity) + self._semantic_score(query_profile, entry)
            ranked.append((score, entry))

        ranked.sort(key=lambda item: item[0], reverse=True)
        unique_entries = []
        seen_timestamps = set()
        for score, entry in ranked:
            if score <= 0:
                continue
            stamp = entry.get("timestamp")
            if stamp in seen_timestamps:
                continue
            seen_timestamps.add(stamp)
            unique_entries.append(entry)
            if len(unique_entries) >= limit:
                break

        return unique_entries

    def search_similar_responses(self, query, query_profile=None, limit=3):

        query_vector = self.embed_query(query)
        if query_vector is None:
            return self._fallback_retrieve(query, query_profile=query_profile, limit=limit)

        return self.retrieve(query, query_profile=query_profile, limit=limit) 
