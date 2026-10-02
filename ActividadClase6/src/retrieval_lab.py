"""Laboratorio de retrieval semántico y evaluación de resultados.

El módulo carga documentos Markdown, los divide en chunks, genera embeddings,
los indexa en Qdrant y calcula métricas sobre un conjunto de consultas.
También puede usarse desde una notebook o desde la línea de comandos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "clase06_retrieval"
ACTIVITY_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_CONFIG: dict[str, Any] = {
    "chunking": {"chunk_size_words": 70, "overlap_words": 0},
    "retrieval": {"top_k": 3, "filters": {}, "score_threshold": None},
}


# ---------------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------------

def parse_scalar(value: str) -> Any:
    """Convierte un valor de frontmatter o CLI a bool, None, int o float.

    Si el texto no representa ninguno de esos tipos, lo devuelve como string.
    """
    value = value.strip().strip('"')
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Separa el frontmatter simple de un Markdown y devuelve metadatos y cuerpo.

    El parser admite pares ``clave: valor`` entre delimitadores ``---``. Si el
    documento no tiene un bloque válido, devuelve metadatos vacíos y el texto
    original completo.
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, Any] = {}
    end = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            end = i
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = parse_scalar(value)

    if end is None:
        return {}, text

    body = "\n".join(lines[end + 1 :]).strip()
    return metadata, body


SECTION_RE = re.compile(r"^##\s+\[section_id:\s*([^\]]+)\]\s*(.+?)\s*$")


def parse_sections(body: str) -> list[dict[str, str]]:
    """Extrae secciones H2 que declaran un ``section_id`` estable.

    Cada elemento contiene el identificador, el título y el texto de la sección.
    El formato esperado es ``## [section_id: ID] Título``.
    """
    sections: list[dict[str, str]] = []
    current_id = None
    current_title = None
    current_lines: list[str] = []

    def flush() -> None:
        """Guarda la sección acumulada y reinicia el estado del parser."""
        nonlocal current_id, current_title, current_lines
        if current_id:
            sections.append(
                {
                    "section_id": current_id,
                    "section_title": current_title or "",
                    "text": "\n".join(current_lines).strip(),
                }
            )
        current_id = None
        current_title = None
        current_lines = []

    for line in body.splitlines():
        match = SECTION_RE.match(line)
        if match:
            flush()
            current_id = match.group(1).strip()
            current_title = match.group(2).strip()
        elif current_id:
            current_lines.append(line)

    flush()
    return sections


def chunk_words(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    """Divide un texto por palabras en chunks con solapamiento configurable."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words debe ser > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words debe ser >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words debe ser menor que chunk_size_words")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def load_corpus(corpus_dir: Path, chunk_size_words: int, overlap_words: int) -> list[dict[str, Any]]:
    """Carga los Markdown del corpus y crea los puntos que se indexarán.

    Cada punto posee un UUID determinista y un payload con el texto del chunk,
    sus metadatos documentales y el ``section_id`` usado durante la evaluación.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del corpus: {corpus_dir}")

    chunks: list[dict[str, Any]] = []

    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        sections = parse_sections(body)

        for section in sections:
            section_chunks = chunk_words(
                section["text"],
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )
            for idx, chunk_text in enumerate(section_chunks):
                stable_name = (
                    f'{metadata.get("document_id", path.stem)}|'
                    f'{section["section_id"]}|{idx}|{chunk_text}'
                )
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_name))

                payload = {
                    **metadata,
                    "source": path.name,
                    "section_id": section["section_id"],
                    "section_title": section["section_title"],
                    "chunk_index": idx,
                    "text": chunk_text,
                }
                chunks.append({"id": point_id, "payload": payload})

    return chunks



# ---------------------------------------------------------------------------
# Exploración pedagógica del corpus
# ---------------------------------------------------------------------------

def corpus_catalog(activity_dir: Path) -> list[dict[str, Any]]:
    """Devuelve una fila por documento para explorar el corpus antes de indexar."""
    corpus_dir = Path(activity_dir) / "corpus"
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del corpus: {corpus_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        sections = parse_sections(body)
        rows.append({
            "document_id": metadata.get("document_id", path.stem),
            "title": metadata.get("title", path.stem),
            "department": metadata.get("department"),
            "topic": metadata.get("topic"),
            "document_type": metadata.get("document_type"),
            "active": metadata.get("active"),
            "version": metadata.get("version"),
            "effective_from": metadata.get("effective_from"),
            "owner": metadata.get("owner"),
            "audience": metadata.get("audience"),
            "country": metadata.get("country"),
            "confidentiality": metadata.get("confidentiality"),
            "n_sections": len(sections),
            "n_words": len(body.split()),
            "source": path.name,
            "content_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10],
        })
    return rows


def document_preview(activity_dir: Path, document_id: str) -> dict[str, Any]:
    """Carga un documento lógico y devuelve metadata + secciones para inspección."""
    corpus_dir = Path(activity_dir) / "corpus"
    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        current_id = metadata.get("document_id", path.stem)
        if current_id == document_id:
            return {
                "source": path.name,
                "metadata": metadata,
                "sections": parse_sections(body),
            }
    raise KeyError(f"No se encontró document_id={document_id!r}")


def inspect_case_results(
    retriever: "DocumentRetriever",
    data_dir: Path,
    case_id: str,
    top_k: int,
    where: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ejecuta una query del golden dataset y devuelve un ranking legible."""
    case = next((c for c in load_cases(Path(data_dir)) if c["case_id"] == case_id), None)
    if case is None:
        raise KeyError(f"No existe el caso {case_id}")

    results, latency_ms = retriever.search(
        query=case["query"],
        top_k=top_k,
        where=where,
        score_threshold=score_threshold,
    )

    expected = set(case["relevant_section_ids"])
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        m = result.metadata
        rows.append({
            "rank": rank,
            "relevante_ground_truth": "✅" if m.get("section_id") in expected else "❌",
            "score": round(result.score, 4),
            "section_id": m.get("section_id"),
            "document_id": m.get("document_id"),
            "title": m.get("title"),
            "department": m.get("department"),
            "topic": m.get("topic"),
            "active": m.get("active"),
            "version": m.get("version"),
            "source": m.get("source"),
            "text": result.text,
        })

    case_info = {
        **case,
        "latency_ms": latency_ms,
    }
    return case_info, rows


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Resultado normalizado de una búsqueda semántica en Qdrant."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class DocumentRetriever:
    """Administra el modelo de embeddings y el índice vectorial local."""

    def __init__(
        self,
        data_dir: Path,
        model_name: str = DEFAULT_MODEL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Inicializa el modelo y abre la base Qdrant de la actividad.

        ``data_dir`` debe ser la carpeta que contiene ``corpus/``; se conserva
        el nombre del parámetro por compatibilidad con la notebook original.
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.collection_name = collection_name

        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=str(self.data_dir / ".qdrant_clase06"))

    def _build_filter(self, where: dict[str, Any] | None) -> models.Filter | None:
        """Traduce un diccionario de igualdad a un filtro ``must`` de Qdrant."""
        if not where:
            return None

        conditions = []
        for key, value in where.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=conditions)

    def close(self) -> None:
        """Cierra Qdrant y libera los archivos de la base vectorial local."""
        self.client.close()

    def rebuild_index(self, chunk_size_words: int, overlap_words: int) -> int:
        """Reconstruye la colección vectorial y devuelve los chunks indexados."""
        chunks = load_corpus(
            self.data_dir / "corpus",
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        if not chunks:
            raise RuntimeError("No se generaron chunks.")

        texts = [c["payload"]["text"] for c in chunks]
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dimension = int(vectors.shape[1])

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

        points = [
            models.PointStruct(
                id=chunk["id"],
                vector=vector.tolist(),
                payload=chunk["payload"],
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 3,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Busca los chunks más similares y devuelve resultados y latencia.

        La latencia medida corresponde a la consulta en Qdrant; la generación
        del embedding de la consulta queda fuera para poder analizar retrieval.
        """
        if not query.strip():
            raise ValueError("query no puede estar vacía")
        if top_k <= 0:
            raise ValueError("top_k debe ser > 0")

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        qfilter = self._build_filter(where)

        start = time.perf_counter()

        # API moderna de qdrant-client.
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qfilter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            # Compatibilidad con versiones anteriores.
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qfilter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

        latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                SearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    text=text,
                    metadata=payload,
                )
            )

        return results, latency_ms


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """Calcula un percentil mediante interpolación lineal."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p debe estar entre 0 y 1")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def evaluate_case(
    retrieved: list[SearchResult],
    relevant_sections: set[str],
    k: int,
) -> dict[str, float]:
    """Calcula Precision@k, Recall@k, RR, Hit@1 y abstención de un caso."""
    if k <= 0:
        raise ValueError("k debe ser > 0")

    retrieved_k = retrieved[:k]
    retrieved_sections = [r.metadata.get("section_id") for r in retrieved_k]

    if not relevant_sections:
        # Las queries sin respuesta se evalúan aparte como abstención.
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "rr": float("nan"),
            "hit_at_1": float("nan"),
            "abstained": 1.0 if len(retrieved_k) == 0 else 0.0,
        }

    relevant_hits = [
        section_id in relevant_sections
        for section_id in retrieved_sections
    ]

    precision = sum(relevant_hits) / k

    found_unique = {
        section_id
        for section_id in retrieved_sections
        if section_id in relevant_sections
    }
    recall = len(found_unique) / len(relevant_sections)

    rr = 0.0
    for rank, section_id in enumerate(retrieved_sections, start=1):
        if section_id in relevant_sections:
            rr = 1.0 / rank
            break

    hit_at_1 = (
        1.0
        if retrieved_sections and retrieved_sections[0] in relevant_sections
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "rr": rr,
        "hit_at_1": hit_at_1,
        "abstained": 0.0,
    }


def load_cases(data_dir: Path) -> list[dict[str, Any]]:
    """Carga y valida mínimamente los casos desde ``evaluation_cases.json``."""
    cases_path = data_dir / "evaluation_cases.json"
    if not cases_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de evaluación: {cases_path}")

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation_cases.json debe contener una lista no vacía")

    required = {"case_id", "query", "case_type", "relevant_section_ids"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"El caso en posición {index} debe ser un objeto JSON")
        missing = required - case.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Al caso {index} le faltan campos: {names}")

    return cases


def run_evaluation(
    retriever: DocumentRetriever,
    data_dir: Path,
    chunk_size_words: int,
    overlap_words: int,
    top_k: int,
    global_filters: dict[str, Any] | None,
    score_threshold: float | None,
    save_report: bool = True,
) -> dict[str, Any]:
    """Reconstruye el índice, evalúa todos los casos y devuelve métricas.

    Si ``save_report`` es verdadero, además guarda el detalle en JSON y CSV
    dentro de ``data_dir/reports``.
    """
    n_chunks = retriever.rebuild_index(chunk_size_words, overlap_words)
    cases = load_cases(data_dir)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for case in cases:
        relevant = set(case["relevant_section_ids"])
        results, latency_ms = retriever.search(
            query=case["query"],
            top_k=top_k,
            where=global_filters,
            score_threshold=score_threshold,
        )
        latencies.append(latency_ms)

        metrics = evaluate_case(results, relevant, k=top_k)

        rows.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "case_type": case["case_type"],
                "relevant_section_ids": sorted(relevant),
                "retrieved_section_ids": [
                    r.metadata.get("section_id") for r in results
                ],
                "scores": [round(r.score, 4) for r in results],
                "max_score": round(max((r.score for r in results), default=float("nan")), 4),
                "returned_count": len(results),
                "retrieved_sources": [r.metadata.get("source") for r in results],
                "retrieved_document_ids": [r.metadata.get("document_id") for r in results],
                "retrieved_active": [r.metadata.get("active") for r in results],
                "retrieved_versions": [r.metadata.get("version") for r in results],
                "precision_at_k": metrics["precision"],
                "recall_at_k": metrics["recall"],
                "rr": metrics["rr"],
                "hit_at_1": metrics["hit_at_1"],
                "abstained": bool(metrics["abstained"]),
                "latency_ms": latency_ms,
            }
        )

    answerable = [r for r in rows if r["relevant_section_ids"]]
    no_answer = [r for r in rows if not r["relevant_section_ids"]]

    summary = {
        "n_chunks": n_chunks,
        "n_cases": len(rows),
        "answerable_cases": len(answerable),
        "no_answer_cases": len(no_answer),
        "precision_at_k": (
            statistics.mean(r["precision_at_k"] for r in answerable)
            if answerable else float("nan")
        ),
        "recall_at_k": (
            statistics.mean(r["recall_at_k"] for r in answerable)
            if answerable else float("nan")
        ),
        "mrr": (
            statistics.mean(r["rr"] for r in answerable)
            if answerable else float("nan")
        ),
        "hit_at_1": (
            statistics.mean(r["hit_at_1"] for r in answerable)
            if answerable else float("nan")
        ),
        "abstention_accuracy": (
            statistics.mean(1.0 if r["abstained"] else 0.0 for r in no_answer)
            if no_answer else float("nan")
        ),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "config": {
            "model": retriever.model_name,
            "chunk_size_words": chunk_size_words,
            "overlap_words": overlap_words,
            "top_k": top_k,
            "filters": global_filters or {},
            "score_threshold": score_threshold,
        },
    }

    if save_report:
        reports_dir = data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = reports_dir / f"evaluation_{stamp}.json"
        json_path.write_text(
            json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        csv_path = reports_dir / f"evaluation_{stamp}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id", "query", "case_type",
                    "relevant_section_ids", "retrieved_section_ids", "scores",
                    "precision_at_k", "recall_at_k", "rr",
                    "abstained", "latency_ms",
                ],
            )
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["relevant_section_ids"] = json.dumps(out["relevant_section_ids"], ensure_ascii=False)
                out["retrieved_section_ids"] = json.dumps(out["retrieved_section_ids"], ensure_ascii=False)
                out["scores"] = json.dumps(out["scores"])
                writer.writerow(out)

        summary["report_json"] = str(json_path)
        summary["report_csv"] = str(csv_path)

    return {"summary": summary, "cases": rows}


def print_summary(report: dict[str, Any], top_k: int) -> None:
    """Imprime las métricas generales y los peores casos para diagnóstico."""
    s = report["summary"]
    print("\n" + "=" * 72)
    print("RESULTADOS")
    print("=" * 72)
    print(f"Chunks indexados       : {s['n_chunks']}")
    print(f"Precision@{top_k:<12}: {s['precision_at_k']:.3f}")
    print(f"Recall@{top_k:<15}: {s['recall_at_k']:.3f}")
    print(f"MRR                    : {s['mrr']:.3f}")
    print(f"Hit@1                  : {s['hit_at_1']:.3f}")
    print(f"Abstention accuracy    : {s['abstention_accuracy']:.3f}")
    print(f"Latencia media         : {s['latency_mean_ms']:.1f} ms")
    print(f"Latencia p50           : {s['latency_p50_ms']:.1f} ms")
    print(f"Latencia p95           : {s['latency_p95_ms']:.1f} ms")

    worst = sorted(
        [r for r in report["cases"] if r["relevant_section_ids"]],
        key=lambda r: (r["rr"], r["recall_at_k"], r["precision_at_k"]),
    )[:5]

    print("\nPeores casos para diagnosticar:")
    for row in worst:
        print(
            f"- {row['case_id']}: RR={row['rr']:.3f} "
            f"R={row['recall_at_k']:.3f} "
            f"P={row['precision_at_k']:.3f}"
        )
        print(f"  Query      : {row['query']}")
        print(f"  Esperado   : {row['relevant_section_ids']}")
        print(f"  Recuperado : {row['retrieved_section_ids']}")
        print(f"  Scores     : {row['scores']}")

    print("\nQueries sin respuesta:")
    for row in report["cases"]:
        if not row["relevant_section_ids"]:
            status = "OK: abstuvo" if row["abstained"] else "FORZÓ RESULTADOS"
            print(f"- {row['case_id']}: {status} | {row['query']}")

    if "report_json" in s:
        print(f"\nReporte JSON: {s['report_json']}")
        print(f"Reporte CSV : {s['report_csv']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_filter(values: list[str] | None) -> dict[str, Any]:
    """Convierte filtros CLI ``clave=valor`` en un diccionario tipado."""
    result: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Filtro inválido: {item}. Usar clave=valor")
        key, value = item.split("=", 1)
        result[key.strip()] = parse_scalar(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Actividad Clase 6 — Retrieval: medir, diagnosticar y mejorar"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ACTIVITY_DIR,
        help="Carpeta de la actividad (debe contener corpus/ y data/).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)

    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser("baseline", help="Ejecutar configuración baseline.")
    baseline.add_argument("--no-save", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Ejecutar un experimento.")
    evaluate.add_argument("--chunk-size", type=int, default=70)
    evaluate.add_argument("--overlap", type=int, default=0)
    evaluate.add_argument("--top-k", type=int, default=3)
    evaluate.add_argument(
        "--filter",
        action="append",
        help='Filtro global repetible. Ej: --filter active=true',
    )
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.add_argument("--no-save", action="store_true")

    query = sub.add_parser("query", help="Inspeccionar una consulta individual.")
    query.add_argument("text")
    query.add_argument("--chunk-size", type=int, default=70)
    query.add_argument("--overlap", type=int, default=0)
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--filter", action="append")
    query.add_argument("--threshold", type=float, default=None)

    return parser


def resolve_evaluation_dir(activity_dir: Path) -> Path:
    """Encuentra la carpeta que contiene ``evaluation_cases.json``.

    Se acepta tanto la estructura actual ``actividad/data`` como una carpeta
    que contenga el JSON directamente, para mantener compatibilidad.
    """
    nested_data_dir = activity_dir / "data"
    if (nested_data_dir / "evaluation_cases.json").is_file():
        return nested_data_dir
    if (activity_dir / "evaluation_cases.json").is_file():
        return activity_dir
    raise FileNotFoundError(
        "No se encontró evaluation_cases.json ni en "
        f"{nested_data_dir} ni en {activity_dir}"
    )


def load_baseline_config(activity_dir: Path) -> dict[str, Any]:
    """Carga la configuración baseline o usa los valores didácticos por defecto."""
    candidates = [
        activity_dir / "data" / "baseline_config.json",
        activity_dir / "baseline_config.json",
    ]
    for config_path in candidates:
        if config_path.is_file():
            return json.loads(config_path.read_text(encoding="utf-8"))
    return DEFAULT_BASELINE_CONFIG


def main() -> None:
    """Ejecuta el comando solicitado: baseline, evaluación o consulta individual."""
    args = build_parser().parse_args()
    activity_dir = args.data_dir.resolve()

    retriever = DocumentRetriever(
        data_dir=activity_dir,
        model_name=args.model,
    )

    if args.command == "baseline":
        evaluation_dir = resolve_evaluation_dir(activity_dir)
        config = load_baseline_config(activity_dir)
        report = run_evaluation(
            retriever=retriever,
            data_dir=evaluation_dir,
            chunk_size_words=config["chunking"]["chunk_size_words"],
            overlap_words=config["chunking"]["overlap_words"],
            top_k=config["retrieval"]["top_k"],
            global_filters=config["retrieval"]["filters"],
            score_threshold=config["retrieval"]["score_threshold"],
            save_report=not args.no_save,
        )
        print_summary(report, top_k=config["retrieval"]["top_k"])

    elif args.command == "evaluate":
        evaluation_dir = resolve_evaluation_dir(activity_dir)
        filters = parse_filter(args.filter)
        report = run_evaluation(
            retriever=retriever,
            data_dir=evaluation_dir,
            chunk_size_words=args.chunk_size,
            overlap_words=args.overlap,
            top_k=args.top_k,
            global_filters=filters,
            score_threshold=args.threshold,
            save_report=not args.no_save,
        )
        print_summary(report, top_k=args.top_k)

    elif args.command == "query":
        filters = parse_filter(args.filter)
        retriever.rebuild_index(args.chunk_size, args.overlap)
        results, latency_ms = retriever.search(
            query=args.text,
            top_k=args.top_k,
            where=filters,
            score_threshold=args.threshold,
        )
        print(f"\nQuery: {args.text}")
        print(f"Latencia: {latency_ms:.1f} ms\n")
        if not results:
            print("Sin resultados por encima del threshold.")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. score={result.score:.4f} | "
                f"{result.metadata.get('section_id')} | "
                f"{result.metadata.get('source')} | "
                f"active={result.metadata.get('active')}"
            )
            print(f"   {result.text[:350]}")
            print()


if __name__ == "__main__":
    main()
