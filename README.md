# SSSP Benchmark Suite

A simple benchmarking tool for Single-Source Shortest Path (SSSP) algorithms. This tool helps you compare the performance of different SSSP algorithms on various graph types.

It implements the following SSSP algorithms:
- Dijkstra's algorithm
- Bellman-Ford algorithm
- Duan-Mao algorithm (https://arxiv.org/abs/2504.17033)

It implements the following graph types:
- Erdős–Rényi random graph
- Barabási–Albert scale-free network
- 2D grid graph
- Random geometric graph
- LiveJournal social network graph
- Wikipedia Talk graph (directed who-talks-to-whom network)

## Installation

First, install the required dependencies:

```bash
pip install networkx pandas scipy
```

## Graph Generation

Generate different types of directed graphs in edgelist format. Each line in the output file represents an edge as:

```
u v w
```

Where:
- `u`, `v`: 0-based node IDs
- `w`: Edge weight (must be > 0)

### Examples

Generate an Erdős–Rényi random graph:
```bash
python gen_graph_edgelist.py --graph erdos --n 10000 --p 5e-5 --out erdos.edgelist
```

Create a Barabási–Albert scale-free network:
```bash
python gen_graph_edgelist.py --graph barabasi --n 50000 --m_attach 5 --out ba.edgelist
```

Generate a 2D grid graph:
```bash
python gen_graph_edgelist.py --graph grid --rows 200 --cols 200 --out grid.edgelist
```

Create a random geometric graph:
```bash
python gen_graph_edgelist.py --graph geometric --n 10000 --radius 0.02 --out geo.edgelist
```

## Running Benchmarks

### Basic Usage

Compare multiple algorithms with multiple trials:
```bash
python bench_sssp.py --graph erdos --n 20000 --p 5e-5 --trials 3 --algos dmsy,dijkstra,bellman_ford --seed 42 --out results.csv
```

Use a custom graph file:
```bash
python bench_sssp.py --input mygraph.edgelist --source 0 --algos dmsy,dijkstra --trials 5 --out results.csv
```

Benchmark with unit weights on a grid:
```bash
python bench_sssp.py --graph grid --rows 200 --cols 200 --weights unit --trials 1 --algos dijkstra --out grid.csv
```

### External Implementations

You can also benchmark external SSSP implementations:
```bash
python bench_sssp.py --graph erdos --n 100000 --p 5e-5 --algos external --external-cmd "./my_sssp" --no-check --out ext.csv
