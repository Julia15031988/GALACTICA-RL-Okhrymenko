from dataclasses import dataclass


@dataclass(slots=True)
class AsteroidState:
    """Represents an asteroid in 2D space."""

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Asteroid radius must be positive.")
