import pytest
from core.entities.agents import (
    VehicleBaseStation,
    FlyingBaseStation,
    AgentManager
)


# --- FIXTURES ---
@pytest.fixture
def empty_registry():
    return AgentManager()


@pytest.fixture
def populated_registry():
    registry = AgentManager()
    vbs = VehicleBaseStation(id=0, capacity=10, coverage_radius=50.0)
    # FBS tethered to VBS 0
    fbs = FlyingBaseStation(id=1, capacity=5, coverage_radius=20.0, host_vbs_id=0)

    registry.register_vbs(vbs)
    registry.register_fbs(fbs)
    return registry


# --- TESTS ---

def test_vbs_initialization():
    """Verifies that the VehicleBaseStation initializes as a pure state tracker."""
    vbs = VehicleBaseStation(id=0, capacity=10, coverage_radius=50.0)
    assert vbs.current_branch_id == 0
    assert vbs.current_slot_index == 0
    assert vbs.current_coverage_count == 0
    assert vbs.is_at_capacity is False


def test_fbs_tethering_logic(empty_registry):
    """Verifies that registering an FBS automatically tethers it to its host VBS."""
    vbs = VehicleBaseStation(id=0, capacity=10, coverage_radius=50.0)
    fbs = FlyingBaseStation(id=1, capacity=5, coverage_radius=20.0, host_vbs_id=0)

    empty_registry.register_vbs(vbs)
    empty_registry.register_fbs(fbs)

    # The VBS should now have the FBS's integer ID in its tethered list
    assert 1 in empty_registry.vbs_registry[0].tethered_fbs_ids


def test_coverage_efficiency_math():
    """Verifies the individual coverage efficiency calculations."""
    vbs = VehicleBaseStation(id=0, capacity=10, coverage_radius=50.0)

    # 0 coverage
    assert vbs.get_coverage_efficiency() == 0.0

    # 50% coverage
    vbs.current_coverage_count = 5
    assert vbs.get_coverage_efficiency() == 0.5
    assert vbs.is_at_capacity is False

    # 100% coverage
    vbs.current_coverage_count = 10
    assert vbs.get_coverage_efficiency() == 1.0
    assert vbs.is_at_capacity is True


def test_system_efficiency_and_reset(populated_registry):
    """Verifies global system math and the reset function used at every environment step."""
    # Manually set some coverage
    populated_registry.vbs_registry[0].current_coverage_count = 5  # 5/10 capacity
    populated_registry.fbs_registry[1].current_coverage_count = 4  # 4/5 capacity

    # Total covered = 9. Total capacity = 15.
    expected_efficiency = 9 / 15
    assert populated_registry.get_total_efficiency() == expected_efficiency

    # Test the reset function
    populated_registry.reset_coverage()

    assert populated_registry.vbs_registry[0].current_coverage_count == 0
    assert populated_registry.fbs_registry[1].current_coverage_count == 0
    assert populated_registry.get_total_efficiency() == 0.0