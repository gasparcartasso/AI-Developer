---
name: sales-analysis
description: Analiza archivos CSV de ventas, valida inconsistencias, calcula facturación, evolución mensual y top de productos. Usar cuando el pedido menciona ventas, CSV, facturación o análisis comercial tabular.
metadata:
  course-version: "1.0.0"
  course-activation: "csv de ventas|ventas|evolución mensual|productos con mayor facturación|análisis de ventas|facturación"
  course-negative-signals: "recursión|política interna|thread de soporte"
  course-tool-hints: "python_execute_sales_analysis"
  course-resources: "references/metric-definitions.md|scripts/analyze-sales.py|assets/report-template.md"
  course-tags: "sales|csv|analytics"
---

# Sales Analysis

## Workflow

1. Inspeccioná columnas y tipos antes de calcular.
2. Validá `fecha`, `producto`, `cantidad` y `precio_unitario`.
3. Una fila es inválida si la fecha no parsea, falta un campo obligatorio,
   cantidad/precio no son numéricos o alguno es negativo.
4. No imputes ni inventes valores: informá fila y motivo; excluí esas filas
   de las métricas y explicá la exclusión.
5. Leé `references/metric-definitions.md` antes de calcular.
6. Calculá facturación solo sobre filas válidas, agrupá por mes y obtené
   los cinco productos con mayor facturación.
7. Generá un gráfico de evolución mensual.
8. Verificá que las salidas respondan al pedido y que los totales cierren.
9. Al redactar, usá `assets/report-template.md` como guía de estructura.

## Reusable script

`scripts/analyze-sales.py` es un helper determinista del paquete. Que exista
dentro de la Skill no significa que la Skill lo ejecute: el runtime debe
invocar una Tool de ejecución.

## Example

Si el usuario pide analizar un CSV y detectar inconsistencias, primero
validá esquema y filas; recién después calculá métricas con registros válidos.
