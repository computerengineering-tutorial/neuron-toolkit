"""Architecture block specifications and declarative signature engine."""

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.architectures.signature_engine import (
    ArchitectureCatalog,
    BlockSignature,
    SignatureEngine,
)

__all__ = [
    "ArchitectureCatalog",
    "BlockSignature",
    "SignatureEngine",
    "build_monomorphized_op_type",
    "compute_subgraph_fingerprint",
    "ensure_specialized_function_proto",
]
