import json
from typing import Tuple, Hashable
import networkx as nx


class NetworkXRoadEngine:
    def __init__(self):
        self.graph = nx.Graph()

    def load_from_json(self, filepath: str):
        """Loads the entire road network using NetworkX node-link standards."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Instantly reconstructs the topology, nodes, edges, and all attributes
        self.graph = nx.node_link_graph(data)

    def save_to_json(self, filepath: str):
        """Exports the network using NetworkX's native node-link format."""
        data = nx.node_link_data(self.graph)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def set_map_dimension(self, dimensions: list):
        """Stores global metadata directly inside the graph object."""
        self.graph.graph["map_dim"] = dimensions

    def get_map_dimension(self):
        return self.graph.graph.get("map_dim")

    def get_neighbors(self, vertex_id: int) -> list[Hashable]:
        return list(self.graph.neighbors(vertex_id))

    def get_edge_coordinates(self, start_vertex: int, end_vertex: int, traveled: float) -> Tuple[float, float]:
        """Calculates physical (x,y) world coordinates based on edge progress."""
        if not (0.0 <= traveled <= 1.0):
            raise ValueError('"traveled" parameter must be between 0.0 and 1.0')

        edge_data = self.graph.edges[start_vertex, end_vertex]
        start_node = self.graph.nodes[start_vertex]
        end_node = self.graph.nodes[end_vertex]

        if edge_data["type"] == "line":
            x = start_node['x'] + traveled * (end_node['x'] - start_node['x'])
            y = start_node['y'] + traveled * (end_node['y'] - start_node['y'])
            return x, y
        else:
            raise NotImplementedError(f"Edge type {edge_data['type']} not implemented")