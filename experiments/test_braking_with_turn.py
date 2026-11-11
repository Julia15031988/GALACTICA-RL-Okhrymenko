import math

import matplotlib.pyplot as plt

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.physics import apply_spaceship_action


def main():
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        vx=9.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    steps = [0]
    speeds = [9.0]
    positions = [spaceship.x]
    angles = [0.0]

    initial_x = spaceship.x

    print("\n========== BRAKING WITH TURN ==========")

    for step in range(1, 30):

        # Rotate toward 180 degrees without thrust.
        if spaceship.angle < math.pi:
            steering = min(1.0, math.pi - spaceship.angle)
            thrust = 0.0
        else:
            steering = 0.0
            thrust = 1.0

        spaceship = apply_spaceship_action(
            spaceship=spaceship,
            steering=steering,
            thrust=thrust,
            dt=1.0,
        )

        speed = math.hypot(
            spaceship.vx,
            spaceship.vy,
        )

        steps.append(step)
        speeds.append(speed)
        positions.append(spaceship.x)
        angles.append(math.degrees(spaceship.angle))

        print(
            f"Step {step:2d} | "
            f"X: {spaceship.x:.2f} | "
            f"Speed: {speed:.2f} | "
            f"Angle: {math.degrees(spaceship.angle):.1f}"
        )

        if spaceship.vx <= 0:
            break

    print("\n========== RESULTS ==========")
    print(f"Braking steps: {steps[-1]}")
    print(f"Braking distance: {spaceship.x - initial_x:.2f}")
    print(f"Fuel remaining: {spaceship.fuel:.2f}")

    # Speed graph.
    plt.figure(figsize=(9, 5))
    plt.plot(
        steps,
        speeds,
        marker="o",
        color="royalblue",
    )
    plt.title("GALACTICA-RL: Braking With Turn")
    plt.xlabel("Simulation step")
    plt.ylabel("Speed")
    plt.grid(alpha=0.3)

    # Position graph.
    plt.figure(figsize=(9, 5))
    plt.plot(
        steps,
        positions,
        marker="o",
        color="darkorange",
    )
    plt.title("Spaceship Position During Braking")
    plt.xlabel("Simulation step")
    plt.ylabel("X position")
    plt.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
