---
document_id: SEC-REMOTE-2026
title: Seguridad y Acceso Remoto
topic: it
doc_type: procedimiento
language: es
active: true
version: "2026.3"
---

# Seguridad y Acceso Remoto

## [section_id: SEC_VPN] Conexión a la red corporativa
El acceso a sistemas internos desde fuera de las oficinas se realiza mediante NexoVPN. El usuario debe iniciar sesión con su cuenta corporativa y completar autenticación multifactor. La VPN es obligatoria para aplicaciones clasificadas como internas, confidenciales o restringidas.

## [section_id: SEC_DISPOSITIVOS] Dispositivos permitidos
Los sistemas corporativos críticos solo pueden utilizarse desde dispositivos administrados por NexoCorp. No se permite acceder a producción, repositorios privados ni consolas administrativas desde computadoras personales. Los teléfonos personales pueden utilizar aplicaciones móviles aprobadas cuando estén protegidos con bloqueo de pantalla.

## [section_id: SEC_MFA] Problemas con MFA
Si el segundo factor deja de funcionar, el usuario debe ingresar al Portal de Ayuda desde un dispositivo ya autenticado y seleccionar “Restablecer MFA”. Si no posee una sesión activa, debe comunicarse con Mesa de Ayuda para validar su identidad antes del reinicio.

## [section_id: SEC_INCIDENTE] Equipo perdido o robado
La pérdida o robo de un dispositivo corporativo debe reportarse inmediatamente a Mesa de Ayuda y Seguridad de la Información. El equipo técnico puede bloquear credenciales, revocar sesiones y ejecutar borrado remoto cuando corresponda. No se debe esperar a realizar una denuncia policial para iniciar el reporte interno.
