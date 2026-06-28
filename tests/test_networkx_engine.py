import json

import networkx as nx
import pytest

from infrastructure.graph.networkx_engine import NetworkXRoadEngine


# --- FIXTURES ---
@pytest.fixture
def sample_graph_file(tmp_path):
    """
    Creates a temporary JSON file containing a valid graph
    to be used by all tests. Cleans up automatically.
    """
    # Create a native NetworkX graph
    G = nx.Graph()
    G.graph["map_dim"] = [100, 100]

    # Add Nodes (0 is center, 1,2,3 are branches)
    G.add_node(0, x=50.0, y=50.0)
    G.add_node(1, x=50.0, y=10.0)  # Branch 0 (South)
    G.add_node(2, x=20.0, y=80.0)  # Branch 1 (North-West)
    G.add_node(3, x=80.0, y=80.0)  # Branch 2 (North-East)

    # Add Edges (All meeting at 0)
    G.add_edge(0, 1, type="line", params={})
    G.add_edge(0, 2, type="line", params={})
    G.add_edge(0, 3, type="line", params={})

    # Export to JSON
    data = nx.node_link_data(G)
    filepath = tmp_path / "test_graph.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)

    return str(filepath)


@pytest.fixture
def engine(sample_graph_file):
    """Returns a loaded engine ready for testing."""
    eng = NetworkXRoadEngine()
    eng.load_from_json(sample_graph_file)
    return eng


# --- TESTS ---

def test_load_and_topology(engine):
    """Verifies that the graph loads correctly and topology is intact."""
    assert len(engine.graph.nodes) == 4
    assert len(engine.graph.edges) == 3

    # Verify the central junction (Node 0) is connected to exactly 3 branches
    neighbors = engine.get_neighbors(0)
    assert set(neighbors) == {1, 2, 3}

    # Verify leaf nodes are only connected to the center
    assert engine.get_neighbors(1) == [0]


def test_save_and_load_consistency(engine, tmp_path):
    """Verifies that saving and reloading does not mutate the graph data."""
    export_path = tmp_path / "export_test.json"
    engine.save_to_json(str(export_path))

    # Create a second engine and load the exported data
    engine2 = NetworkXRoadEngine()
    engine2.load_from_json(str(export_path))

    # Check if global attributes and node coordinates survived the round trip
    assert engine2.get_map_dimension() == [100, 100]
    assert engine2.graph.nodes[2]['x'] == 20.0


def test_get_edge_coordinates_interpolation(engine):
    """Verifies the linear interpolation math for agent movement."""
    # Test Node 0 to Node 1 (Vertical line from y=50 to y=10)
    # 0% traveled -> exactly at start node
    x, y = engine.get_edge_coordinates(0, 1, 0.0)
    assert (x, y) == (50.0, 50.0)

    # 50% traveled -> exactly in the middle
    x, y = engine.get_edge_coordinates(0, 1, 0.5)
    assert (x, y) == (50.0, 30.0)

    # 100% traveled -> exactly at end node
    x, y = engine.get_edge_coordinates(0, 1, 1.0)
    assert (x, y) == (50.0, 10.0)

    # 25% traveled backward -> 3/4 from 0 to 1
    x, y = engine.get_edge_coordinates(1, 0, 0.25)
    assert (x, y) == (50.0, 20.0)


def test_get_edge_coordinates_bounds(engine):
    """Verifies that the environment securely rejects invalid physics updates."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        engine.get_edge_coordinates(0, 1, -0.1)

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        engine.get_edge_coordinates(0, 1, 1.1)