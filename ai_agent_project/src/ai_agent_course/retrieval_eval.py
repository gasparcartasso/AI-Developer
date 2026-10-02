# Métricas reutilizables e independientes del backend.
# Trabajamos a nivel documento para no contar varias veces chunks del mismo origen.

def unique_document_ids(results, k):
    ids = []

    for result in results:
        if result.document_id not in ids:
            ids.append(result.document_id)

        if len(ids) == k:
            break

    return ids


def precision_at_k(results, relevant_ids, k):
    retrieved = unique_document_ids(results, k)
    if not retrieved:
        return 0.0

    relevant = set(relevant_ids)
    return sum(doc_id in relevant for doc_id in retrieved) / len(retrieved)


def recall_at_k(results, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    retrieved = set(unique_document_ids(results, k))
    return len(retrieved & relevant) / len(relevant)


def hit_rate_at_k(results, relevant_ids, k):
    relevant = set(relevant_ids)
    retrieved = set(unique_document_ids(results, k))
    return float(bool(retrieved & relevant))


def reciprocal_rank(results, relevant_ids):
    relevant = set(relevant_ids)
    seen = set()
    document_rank = 0

    for result in results:
        if result.document_id in seen:
            continue

        seen.add(result.document_id)
        document_rank += 1

        if result.document_id in relevant:
            return 1.0 / document_rank

    return 0.0
