class LLMError(Exception):
    """Error base de la capa de modelos."""


class ProviderConfigurationError(LLMError):
    """Configuración o credenciales inválidas. No se reintenta."""


class InvalidRequestError(LLMError):
    """Input inválido. No se reintenta."""


class InvalidProviderResponseError(LLMError):
    """El proveedor respondió, pero no cumplió el contrato."""


class TransientProviderError(LLMError):
    """Falla temporal del proveedor. Puede reintentarse."""


class RateLimitError(TransientProviderError):
    """El proveedor rechazó temporalmente por límite de uso."""


class ProviderTimeoutError(TransientProviderError):
    """La llamada excedió el tiempo máximo permitido."""


class PartialStreamError(LLMError):
    """El stream falló después de emitir contenido; reintentar podría duplicarlo."""
