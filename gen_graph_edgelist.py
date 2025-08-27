#!/usr/bin/env python3
"""
Graph Generator for SSSP Benchmark
Creates directed graphs and writes them in edgelist format:
    u v w
where u,v are integers (0-based node IDs), w is edge weight > 0.

Examples:
  python gen_graph_edgelist.py --graph erdos --n 10000 --p 5e-5 --out erdos.edgelist
  python gen_graph_edgelist.py --graph barabasi --n 50000 --m_attach 5 --out ba.edgelist
  python gen_graph_edgelist.py --graph grid --rows 200 --cols 200 --out grid.edgelist
  python gen_graph_edgelist.py --graph geometric --n 10000 --radius 0.02 --out geo.edgelist
"""

import argparse
import random
import networkx as nx

def assign_weights(g, weights, rng):
    if weights == "unit":
        for u, v in g.edges():
            g[u][v]["weight"] = 1.0
    elif weights == "lo":
        for u, v in g.edges():
            g[u][v]["weight"] = rng.uniform(1.0, 2.0)
    elif weights == "hi":
        for u, v in g.edges():
            g[u][v]["weight"] = rng.uniform(0.5, 100.0)
    else:  # custom
        for u, v in g.edges():
            g[u][v]["weight"] = rng.uniform(1e-3, 1e3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["erdos", "barabasi", "grid", "geometric"], default="erdos")
    parser.add_argument("--n", type=int, default=1000, help="number of nodes (erdos, barabasi, geometric)")
    parser.add_argument("--p", type=float, default=5e-3, help="edge probability (erdos)")
    parser.add_argument("--m_attach", type=int, default=3, help="edges to attach (barabasi)")
    parser.add_argument("--rows", type=int, default=50, help="grid rows")
    parser.add_argument("--cols", type=int, default=50, help="grid cols")
    parser.add_argument("--radius", type=float, default=0.05, help="geometric radius")
    parser.add_argument("--weights", choices=["unit","lo","hi","custom"], default="lo")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="mygraph.edgelist")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.graph == "erdos":
        g = nx.gnp_random_graph(args.n, args.p, seed=args.seed, directed=True)
    elif args.graph == "barabasi":
        ug = nx.barabasi_albert_graph(args.n, args.m_attach, seed=args.seed)
        g = ug.to_directed()
    elif args.graph == "grid":
        g = nx.grid_2d_graph(args.rows, args.cols, create_using=nx.DiGraph)
        mapping = {(r,c): r*args.cols+c for r,c in g.nodes()}
        g = nx.relabel_nodes(g, mapping)
    elif args.graph == "geometric":
        ug = nx.random_geometric_graph(args.n, args.radius, seed=args.seed)
        g = nx.DiGraph()
        g.add_nodes_from(ug.nodes)
        for u,v in ug.edges:
            if rng.random() < 0.8: g.add_edge(u,v)
            if rng.random() < 0.8: g.add_edge(v,u)
    else:
        raise ValueError("Unknown graph type")

    assign_weights(g, args.weights, rng)

    with open(args.out, "w") as f:
        for u,v,data in g.edges(data=True):
            f.write(f"{u} {v} {data['weight']}\n")

    print(f"Wrote {g.number_of_edges()} edges on {g.number_of_nodes()} nodes to {args.out}")

if __name__ == "__main__":
    main()
