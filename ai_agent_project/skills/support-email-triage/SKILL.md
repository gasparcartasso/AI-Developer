---
name: support-email-triage
description: Analiza emails y threads de soporte, identifica riesgo, busca evidencia interna y prepara una respuesta trazable. Usar para soporte, email, thread, incidentes técnicos, portal, facturas o contraseñas.
metadata:
  course-version: "1.0.0"
  course-activation: "email|thread|soporte|factura|contraseña|portal|incidente"
  course-negative-signals: "vacaciones|reintegro|licencia"
  course-tool-hints: "gmail_read_thread|search_support_kb|lookup_service_status|gmail_create_draft"
  course-resources: "references/triage-checklist.md"
  course-tags: "support|email|triage"
---

# Support Email Triage

## Workflow

1. Leé el thread antes de proponer una respuesta.
2. Extraé hechos observables y preguntas abiertas.
3. Buscá evidencia en la KB interna cuando la solución dependa de procedimiento.
4. Si el incidente bloquea una operación crítica, elevá prioridad.
5. No inventes estado actual del servicio: consultalo con una Tool si hace falta.
6. Un draft es una propuesta; no equivale a enviar el email.

## Example

Si el portal no deja cargar facturas y bloquea el cierre, clasificá
prioridad, recuperá procedimiento interno y pedí la información faltante.
