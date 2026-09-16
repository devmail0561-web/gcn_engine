"""Frontend bridge : texte brut → UDRepresentation via gcn-cli subprocess."""
from .bridge import reps_from_text, GCNBridgeError

__all__ = ["reps_from_text", "GCNBridgeError"]
