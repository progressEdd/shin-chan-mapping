import numpy as np
from typing import List, Dict, Optional, Sequence, Union

try:
    import ollama  # optional; only needed if you pass `text` instead of `query_vector`
except Exception:
    ollama = None


def _as_unit_vec(vec_like: Union[Sequence[float], Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """
    Accepts [d] or [[d]] and returns a float32 unit vector (cosine-normalized).
    Returns None if it can't be coerced.
    """
    if vec_like is None:
        return None
    # squeeze [[...]] -> [...]
    if isinstance(vec_like, (list, tuple)) and vec_like and isinstance(vec_like[0], (list, tuple, np.ndarray)):
        vec_like = vec_like[0]
    try:
        v = np.asarray(vec_like, dtype=np.float32)
    except Exception:
        return None
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all():
        return None
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return None
    return v / n


def _detect_embedding_key(rec: Dict) -> Optional[str]:
    """
    Guess which key holds the embedding. Prioritize common names; otherwise
    pick the first list/array of numbers that looks like a vector.
    """
    candidates = [
        "embedding", "title_embedding", "vector", "embedding_vec",
        "emb", "repr", "embedding_vector"
    ]
    for k in candidates:
        if k in rec:
            v = _as_unit_vec(rec[k])
            if v is not None:
                return k
    # fallback: scan all keys
    for k, v in rec.items():
        if _as_unit_vec(v) is not None:
            return k
    return None


def _safe_get(rec: Dict, key: str, default=None):
    return rec.get(key, default)


def vector_search(
    records: List[Dict],
    *,
    # You can pass EITHER text+model OR a precomputed query_vector
    text: Optional[str] = None,
    model: Optional[str] = None,
    query_vector: Optional[Sequence[float]] = None,
    top_k: int = 10,
    embedding_key: Optional[str] = None,       # if None, auto-detect from the first record
    return_fields: Optional[Sequence[str]] = None,  # if None, return all fields except the embedding
    min_dim: int = 1,                           # ignore embeddings with dim < min_dim
) -> List[Dict]:
    """
    Generic cosine-similarity search over any list of dicts that include an embedding.

    Parameters
    ----------
    records : list of dict
        Each dict should contain an embedding under some key. Key can vary
        (e.g., 'embedding', 'title_embedding', etc.).
    text : str, optional
        Text to embed using `model` via ollama.
    model : str, optional
        Embedding model name for ollama (required if `text` is provided).
    query_vector : sequence of float, optional
        Precomputed query vector. If provided, `text` is ignored.
    top_k : int
        Number of results to return.
    embedding_key : str, optional
        Force the name of the embedding key. If None, auto-detected from the first feasible record.
    return_fields : list[str], optional
        Which fields to return from each record. If None, returns all except the embedding field.
    min_dim : int
        Throw away records whose embedding dimension is < min_dim.

    Returns
    -------
    list of dict with fields of the source record (filtered by `return_fields`) + `score`.
    """
    if not records:
        return []

    # Determine embedding key if not provided
    if embedding_key is None:
        embedding_key = _detect_embedding_key(records[0]) or _detect_embedding_key(
            next((r for r in records if r is not None), {})
        )
        if embedding_key is None:
            raise ValueError("Could not detect an embedding key in the records. Pass `embedding_key=` explicitly.")

    # Build matrix of unit vectors and keep an index map
    matrix = []
    kept = []  # indices of records kept
    dims = None

    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or embedding_key not in rec:
            continue
        v = _as_unit_vec(rec[embedding_key])
        if v is None or v.size < min_dim:
            continue
        if dims is None:
            dims = v.size
        # Filter out dimensionality mismatches
        if v.size != dims:
            continue
        matrix.append(v)
        kept.append(i)

    if not matrix:
        raise ValueError(f"No valid embeddings found under key '{embedding_key}' (dimension mismatch or invalid data).")

    G = np.vstack(matrix)  # (N, d)

    # Prepare the query vector
    if query_vector is not None:
        q = _as_unit_vec(query_vector)
        if q is None or q.size != G.shape[1]:
            raise ValueError("`query_vector` is invalid or has mismatched dimensionality.")
    else:
        if text is None:
            raise ValueError("Provide either `query_vector` or (`text` and `model`).")
        if model is None:
            raise ValueError("`model` must be provided when using `text`.")
        if ollama is None:
            raise RuntimeError("`ollama` is not available to embed text. Install/import it or pass `query_vector`.")
        # Ollama returns {"embeddings": [[...]]} (model-dependent)
        q_raw = ollama.embed(model=model, input=text).embeddings
        q = _as_unit_vec(q_raw)
        if q is None or q.size != G.shape[1]:
            raise ValueError("Embedded query has invalid shape or mismatched dimensionality.")

    # Cosine similarity = dot product of unit vectors
    scores = G @ q

    # Top-k
    k = int(max(1, min(top_k, scores.size)))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    # Build results
    out = []
    for local_i in idx:
        global_i = kept[local_i]
        rec = records[global_i]
        row = {}

        # Decide which fields to return
        if return_fields is None:
            # everything except the embedding field
            for kf, vf in rec.items():
                if kf != embedding_key:
                    row[kf] = vf
        else:
            for kf in return_fields:
                row[kf] = _safe_get(rec, kf)

        row["score"] = float(scores[local_i])
        out.append(row)

    return out
