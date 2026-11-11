from types import SimpleNamespace

from galactica.environment.maneuvering_physics import (
    apply_maneuvering_action,
)


def create_ship():
    return SimpleNamespace(
        x=10.0,
        y=50.0,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )


def show_state(label, ship):
    print(f"\n{label}")
    print(f"Position: ({ship.x:.2f}, {ship.y:.2f})")
    print(f"Velocity: ({ship.vx:.2f}, {ship.vy:.2f})")
    print(f"Fuel: {ship.fuel:.2f}")


def main():
    # TEST 1: Forward acceleration
    ship = create_ship()

    for _ in range(5):
        apply_maneuvering_action(
            ship, 0.0, 1.0, 0.0, 0.0
        )

    show_state("TEST 1: FORWARD", ship)
    assert ship.vx == 5.0

    # TEST 2: Reverse braking
    for _ in range(5):
        apply_maneuvering_action(
            ship, 0.0, 0.0, 1.0, 0.0
        )

    show_state("TEST 2: BRAKING", ship)
    assert abs(ship.vx) < 1e-9

    # TEST 3: Lateral movement
    ship = create_ship()

    apply_maneuvering_action(
        ship, 0.0, 0.0, 0.0, 1.0
    )

    show_state("TEST 3: LATERAL", ship)
    assert abs(ship.vy - 0.6) < 1e-9

    # TEST 4: Inertia
    ship = create_ship()
    ship.vx = 3.0

    apply_maneuvering_action(
        ship, 0.0, 0.0, 0.0, 0.0
    )

    show_state("TEST 4: INERTIA", ship)
    assert ship.vx == 3.0
    assert ship.x == 13.0

    # TEST 5: No fuel
    ship = create_ship()
    ship.fuel = 0.0

    apply_maneuvering_action(
        ship, 0.0, 1.0, 0.0, 0.0
    )

    show_state("TEST 5: NO FUEL", ship)
    assert ship.vx == 0.0
    assert ship.fuel == 0.0

    print("\nALL MANEUVERING PHYSICS TESTS PASSED!")


if __name__ == "__main__":
    main()
