"""Filament compatibility hints for parts printed against each other."""

from __future__ import annotations

# Pairs known to bond poorly when printed against each other (order-free).
# Conservative: only well-known combinations; anything unknown gets no warning.
_POOR = {
    frozenset({"PLA", "PETG"}): "PLA and PETG barely bond (they're used as support interface for each other)",
    frozenset({"PETG", "ABS"}): "PETG and ABS bond poorly",
    frozenset({"PETG", "ASA"}): "PETG and ASA bond poorly",
    frozenset({"PLA", "ABS"}): "PLA and ABS bond poorly and print at very different temperatures",
    frozenset({"PLA", "ASA"}): "PLA and ASA bond poorly and print at very different temperatures",
    frozenset({"PLA", "PA"}): "PLA and nylon bond poorly",
    frozenset({"PETG", "PA"}): "PETG and nylon bond poorly",
}


def _family(t: str) -> str:
    t = t.upper().strip()
    for fam in ("PETG", "PLA", "ABS", "ASA", "PVB", "TPU", "PA", "PC"):
        if t.startswith(fam):
            return fam
    return t


def warnings(project, pairs: set[tuple[int, int]]) -> list[str]:
    """Warnings for filament pairs (1-based) that will touch in the print."""
    out = []
    for a, b in sorted(pairs):
        ta, tb = _family(project.filament_type(a)), _family(project.filament_type(b))
        msg = _POOR.get(frozenset({ta, tb}))
        if msg and ta != tb:
            out.append(f"filaments {a} ({project.filament_type(a)}) and {b} ({project.filament_type(b)}): {msg}")
    return out
