import math

import matplotlib.pyplot as plt

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.physics import apply_spaceship_action


def main():
    # Initial spaceship state.
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        vx=9.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    initial_x = spaceship.x
    initial_speed = math.hypot(spaceship.vx, spaceship.vy)

    # Turn the engine against the direction of movement.
    # This experiment assumes the ship is already facing backward.
    spaceship = SpaceshipState(
        x=spaceship.x,
        y=spaceship.y,
        radius=spaceship.radius,
        vx=spaceship.vx,
        vy=spaceship.vy,
        angle=math.pi,
        fuel=spaceship.fuel,
    )

    steps = [0]
    speeds = [initial_speed]
    positions = [spaceship.x]

    print("\n========== GALACTICA BRAKING TEST ==========")
    print(f"Initial speed: {initial_speed:.2f}")

    for step in range(1, 20):
        spaceship = apply_spaceship_action(
            spaceship=spaceship,
            steering=0.0,
            thrust=1.0,
            dt=1.0,
        )

        speed = math.hypot(spaceship.vx, spaceship.vy)

        steps.append(step)
        speeds.append(speed)
        positions.append(spaceship.x)

        print(
            f"Step {step:2d} | "
            f"X: {spaceship.x:.2f} | "
            f"Speed: {speed:.2f}"
        )

        if spaceship.vx <= 0:
            break

    braking_distance = spaceship.x - initial_x

    print("\n========== RESULTS ==========")
    print(f"Initial speed: {initial_speed:.2f}")
    print(f"Final speed: {speeds[-1]:.2f}")
    print(f"Braking steps: {steps[-1]}")
    print(f"Braking distance: {braking_distance:.2f}")
    print(f"Fuel remaining: {spaceship.fuel:.2f}")

    # Plot speed during braking.
    plt.figure(figsize=(9, 5))

    plt.plot(
        steps,
        speeds,
        marker="o",
        linewidth=2,
        color="royalblue",
    )

    plt.title("GALACTICA-RL: Spaceship Braking Test")
    plt.xlabel("Simulation step")
    plt.ylabel("Speed")
    plt.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
