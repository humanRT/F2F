"""Flushed elapsed-time progress for interactive PowerShell runs."""
from time import monotonic

_started = monotonic()


def progress(stage, message):
    print(f"[{monotonic()-_started:7.1f}s] {stage}: {message}", flush=True)


def pairing_progress(iteration, total, valid, count, movement_m):
    change = f"{movement_m*1000:.3f} mm" if movement_m != float('inf') else "unavailable"
    progress("Pairing", f"iteration {iteration}/{total}; {valid}/{count} accepted; max movement {change}")
