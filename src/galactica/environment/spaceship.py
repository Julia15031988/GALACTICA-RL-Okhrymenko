from dataclasses import dataclass


@dataclass(slots=True)
class SpaceshipState:
    """Represents the current state of a spaceship in 2D space."""

    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0
    fuel: float = 100.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Spaceship radius must be positive.")

        if self.fuel < 0:
            raise ValueError("Fuel cannot be negative.")
