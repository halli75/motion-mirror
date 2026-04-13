from dataclasses import dataclass


@dataclass(slots=True)
class GenerationRequest:
    backend: str = "wan-move-14b"
    resolution: str = "832x480"
    frames: int = 81
