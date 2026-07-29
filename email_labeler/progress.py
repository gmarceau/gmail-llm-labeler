"""Shared progress bar for long-running CLI operations."""

from progress.bar import Bar


class SmartBar(Bar):
    """Progress bar with a seeded initial ETA for the first few iterations."""

    def __init__(self, message: str, max: int, initial_eta_seconds: float):
        self._initial_eta_seconds = initial_eta_seconds
        super().__init__(message, max=max, suffix="%(index)d/%(max)d ETA %(eta_hms)s")

    @property
    def eta(self) -> float:
        if self.index <= 1:
            return self._initial_eta_seconds
        return super().eta  # type: ignore[misc]

    @property
    def eta_hms(self) -> str:
        s = self.eta
        if s >= 3600:
            return f"{int(s // 3600)}h{int((s % 3600) // 60)}m"
        elif s >= 60:
            return f"{int(s // 60)}m{int(s % 60)}s"
        else:
            return f"{int(s)}s"
