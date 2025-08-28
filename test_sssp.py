"""Tests for the SSSP benchmark suite."""
import networkx as nx
from bench_sssp import (
    run_dijkstra,
    run_bellman_ford,
    run_dmsy,
    build_erdos,
    build_barabasi,
    build_grid,
    build_geometric,
)


def test_dijkstra():
    print("Test Dijkstra's algorithm on a small graph.")
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)])
    dist = run_dijkstra(g, 0)
    assert dist == {0: 0, 1: 1, 2: 3, 3: 4}
test_dijkstra()

def test_bellman_ford():
    print("Test Bellman-Ford algorithm on a small graph.")
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)])
    dist = run_bellman_ford(g, 0)
    assert dist == {0: 0, 1: 1, 2: 3, 3: 4}
test_bellman_ford()


def test_dmsy():
    print("Test Duan-Mao algorithm on a small graph.")
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)])
    dist = run_dmsy(g, 0)
    assert dist == {0: 0, 1: 1, 2: 3, 3: 4}
test_dmsy()

def test_graph_generators():
    print("Test that graph generators produce valid graphs.")
    # Test Erdos-Renyi graph
    g = build_erdos(10, 0.3, 42, "unit")
    assert len(g) == 10
    
    # Test Barabasi-Albert graph
    g = build_barabasi(10, 2, 42, "unit")
    assert len(g) == 10
    
    # Test Grid graph
    g = build_grid(3, 3, "unit")
    assert len(g) == 9  # 3x3 grid has 9 nodes
    
    # Test Geometric graph
    g = build_geometric(10, 0.5, 42, "unit")
    assert len(g) == 10
test_graph_generators() 