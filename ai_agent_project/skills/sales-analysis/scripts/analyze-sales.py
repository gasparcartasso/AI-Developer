import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("ventas.csv")
RESULT = Path("analysis_result.json")
REPORT = Path("informe_ventas.md")
CHART = Path("evolucion_mensual.png")

raw = pd.read_csv(INPUT)
required = ["fecha", "producto", "cantidad", "precio_unitario"]
missing_columns = [c for c in required if c not in raw.columns]
if missing_columns:
    raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

work = raw.copy()
work["_fecha"] = pd.to_datetime(work["fecha"], errors="coerce")
work["_cantidad"] = pd.to_numeric(work["cantidad"], errors="coerce")
work["_precio"] = pd.to_numeric(work["precio_unitario"], errors="coerce")

inconsistencies = []
valid_mask = []
for idx, row in work.iterrows():
    reasons = []
    if pd.isna(row["_fecha"]):
        reasons.append("fecha inválida")
    if pd.isna(row["producto"]) or not str(row["producto"]).strip():
        reasons.append("producto faltante")
    if pd.isna(row["_cantidad"]):
        reasons.append("cantidad faltante/no numérica")
    elif row["_cantidad"] < 0:
        reasons.append("cantidad negativa")
    if pd.isna(row["_precio"]):
        reasons.append("precio_unitario faltante/no numérico")
    elif row["_precio"] < 0:
        reasons.append("precio_unitario negativo")

    is_valid = not reasons
    valid_mask.append(is_valid)
    if reasons:
        inconsistencies.append({
            "fila_csv": int(idx + 2),  # +1 por índice base 0 y +1 por encabezado
            "motivos": reasons,
            "valores": {
                "fecha": None if pd.isna(row["fecha"]) else str(row["fecha"]),
                "producto": None if pd.isna(row["producto"]) else str(row["producto"]),
                "cantidad": None if pd.isna(row["cantidad"]) else float(row["cantidad"]),
                "precio_unitario": None if pd.isna(row["precio_unitario"]) else float(row["precio_unitario"]),
            },
        })

valid = work.loc[valid_mask].copy()
valid["facturacion"] = valid["_cantidad"] * valid["_precio"]
valid["mes"] = valid["_fecha"].dt.to_period("M").astype(str)

monthly = (
    valid.groupby("mes", as_index=False)["facturacion"]
    .sum()
    .sort_values("mes")
)
top_products = (
    valid.groupby("producto", as_index=False)["facturacion"]
    .sum()
    .sort_values(["facturacion", "producto"], ascending=[False, True])
    .head(5)
)

# Verificación simple: el total mensual debe cerrar con el total de filas válidas.
total_valid = float(valid["facturacion"].sum())
assert abs(float(monthly["facturacion"].sum()) - total_valid) < 1e-9

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(monthly["mes"], monthly["facturacion"], marker="o")
ax.set_title("Evolución mensual de facturación")
ax.set_xlabel("Mes")
ax.set_ylabel("Facturación")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(CHART, dpi=140)
plt.close(fig)

result = {
    "rows_total": int(len(raw)),
    "rows_valid": int(len(valid)),
    "rows_invalid": int(len(inconsistencies)),
    "total_facturacion_valida": total_valid,
    "monthly": [
        {"mes": r["mes"], "facturacion": float(r["facturacion"])}
        for r in monthly.to_dict(orient="records")
    ],
    "top_products": [
        {"producto": r["producto"], "facturacion": float(r["facturacion"])}
        for r in top_products.to_dict(orient="records")
    ],
    "inconsistencies": inconsistencies,
    "treatment": (
        "Las filas inválidas se informan y se excluyen de las métricas; "
        "no se imputan ni corrigen valores silenciosamente."
    ),
}
RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# Informe breve de ventas",
    "",
    "## Evolución mensual",
]
for item in result["monthly"]:
    lines.append(f"- {item['mes']}: {item['facturacion']:.2f}")
lines += ["", "## Top 5 productos"]
for item in result["top_products"]:
    lines.append(f"- {item['producto']}: {item['facturacion']:.2f}")
lines += ["", "## Inconsistencias"]
for item in inconsistencies:
    lines.append(f"- Fila {item['fila_csv']}: {', '.join(item['motivos'])}")
lines += [
    "",
    "## Tratamiento y limitaciones",
    "- No se imputaron datos faltantes ni se corrigieron fechas inválidas.",
    "- Las métricas usan únicamente filas válidas del CSV sintético.",
]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(result, ensure_ascii=False, indent=2))
