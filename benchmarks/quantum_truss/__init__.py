"""Scalable truss benchmarks shared by classical and quantum backends."""

__all__ = ["CASE_SIZES", "generate_case"]


def __getattr__(name):
    # Keep ``python -m benchmarks.quantum_truss.generate_cases`` warning-free.
    if name in __all__:
        from .generate_cases import CASE_SIZES, generate_case

        return {"CASE_SIZES": CASE_SIZES, "generate_case": generate_case}[name]
    raise AttributeError(name)
