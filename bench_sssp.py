#!/usr/bin/env python3
"""
-Requirements:
 - Python 3.9+
 - networkx >= 3.0
-Usage examples:
 python bench_sssp.py --graph erdos --n 20000 --p 5e-5 --trials 3 --algos dijkstra,bellman_ford --seed 42 --out results.csv
 python bench_sssp.py --input mygraph.edgelist --source 0 --algos dijkstra --trials 5 --out results.csv
 python bench_sssp.py --graph barabasi --n 100000 --m_attach 3 --weight hi --algos dijkstra --trials 1 --out ba.csv
 python bench_sssp.py --graph grid --rows 200 --cols 200 --weights unit --trials 1 --algos dijkstra --out grid.csv
 python bench_sssp.py --graph erdos --n 1_000_000 --p 5e-7 --algos dijkstra --trials 1 --out big.csv --no_check
-
-Optional external algorithm:
 Use --external-cmd to benchmark your own executable or script.
 Protocol on stdin:
   First line: n m s
   Next m lines: u v w  zero based nodes  w > 0
 Expected stdout, one of:
   - Single line of n whitespace separated distances
   - Or n lines: "node distance"
 Example:
   --external-cmd "./my_sssp --float32"
-
-Notes:
 - All generated graphs are directed.
 - Weights are positive. Choose --weights unit, lo, mid, hi or --weights custom to set ranges.
 - If you disable checks with --no_check, the script will not compare results with the reference.
"""

import argparse
import csv
import math
import os
import random
import shlex
import subprocess
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple, Optional

# optional deps
try:
    import psutil  # for RSS memory
except Exception:
    psutil = None

try:
    import resource  # Unix peak RSS fallback
except Exception:
    resource = None

try:
    import networkx as nx
except Exception as e:
    print("NetworkX is required. pip install networkx", file=sys.stderr)
    raise

# ------------- Graph builders -------------
DEFAULT_SEED = random.randint(0, 2**32 - 1)

def get_or_generate_seed(seed: Optional[int] = None) -> int:
    """Generate a random seed if none provided and print it for reproducibility."""
    if seed is None:
        seed = DEFAULT_SEED
        print(f"Using generated seed: {seed}")
    return seed

def build_erdos(n: int, p: float, seed: Optional[int], weights: str) -> "nx.DiGraph":
    seed = get_or_generate_seed(seed)
    rng = random.Random(seed)
    g = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    assign_weights(g, weights, rng)
    return g

def build_barabasi(n: int, m_attach: int, seed: Optional[int], weights: str) -> "nx.DiGraph":
    seed = get_or_generate_seed(seed)
    rng = random.Random(seed)
    ug = nx.barabasi_albert_graph(n, m_attach, seed=seed)
    g = ug.to_directed()
    assign_weights(g, weights, rng)
    return g

def build_grid(rows: int, cols: int, weights: str) -> "nx.DiGraph":
    g = nx.grid_2d_graph(rows, cols, create_using=nx.DiGraph)
    mapping = {(r, c): r * cols + c for r, c in g.nodes}
    g = nx.relabel_nodes(g, mapping)
    assign_weights(g, weights, random.Random(get_or_generate_seed()))
    return g

def build_geometric(n: int, radius: float, seed: Optional[int], weights: str) -> "nx.DiGraph":
    seed = get_or_generate_seed(seed)
    rng = random.Random(seed)
    ug = nx.random_geometric_graph(n, radius, seed=seed)
    g = nx.DiGraph()
    g.add_nodes_from(ug.nodes)
    for u, v in ug.edges:
        if rng.random() < 0.8:
            g.add_edge(u, v)
        if rng.random() < 0.8:
            g.add_edge(v, u)
    assign_weights(g, weights, rng)
    return g

def download_file(url: str, path: str) -> None:
    import urllib.request
    import gzip
    import shutil

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url}...")
    temp_path = f"{path}.download"
    urllib.request.urlretrieve(url, temp_path)
    if url.endswith(".gz"):
        with gzip.open(temp_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(temp_path)
    else:
        os.replace(temp_path, path)
    print(f"Downloaded and extracted to {path}")

def build_livejournal(weights: str = "unit") -> "nx.DiGraph":
    print("/===warning: this going to take a while and a lot of memory ===\\")
    import scipy.sparse as sp
    url = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
    path = os.path.join("data", "soc-LiveJournal1.txt")
    if not os.path.exists(path):
        download_file(url, path)
    print("Loading LiveJournal graph...")
    node_set = set()
    edge_count = 0
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            node_set.update([u, v])
            edge_count += 1
    nodes = sorted(node_set)
    node_id = {n: i for i, n in enumerate(nodes)}
    del node_set
    print(f"Processing {len(nodes):,} nodes and ~{edge_count:,} edges...")
    rows, cols = [], []
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            rows.append(node_id[u])
            cols.append(node_id[v])
    n = len(nodes)
    adj_matrix = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
    print("Converting to NetworkX graph...")
    g = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    mapping = {i: n for i, n in enumerate(nodes)}
    g = nx.relabel_nodes(g, mapping)
    print("Finding largest weakly connected component...")
    largest_cc = max(nx.weakly_connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()
    assign_weights(g, weights, random.Random(get_or_generate_seed()))
    return g

def build_wiki_talk(weights: str = "unit") -> "nx.DiGraph":
    import scipy.sparse as sp
    url = "https://snap.stanford.edu/data/wiki-Talk.txt.gz"
    path = os.path.join("data", "wiki-Talk.txt")
    if not os.path.exists(path):
        download_file(url, path)
    print("Loading Wiki-Talk graph...")
    node_set = set()
    edge_count = 0
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            node_set.update([u, v])
            edge_count += 1
    nodes = sorted(node_set)
    node_id = {n: i for i, n in enumerate(nodes)}
    del node_set
    print(f"Processing {len(nodes):,} nodes and ~{edge_count:,} edges...")
    rows, cols = [], []
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            rows.append(node_id[u])
            cols.append(node_id[v])
    n = len(nodes)
    adj_matrix = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
    print("Converting to NetworkX graph...")
    g = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    mapping = {i: n for i, n in enumerate(nodes)}
    g = nx.relabel_nodes(g, mapping)
    print("Finding largest weakly connected component...")
    largest_cc = max(nx.weakly_connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()
    assign_weights(g, weights, random.Random(get_or_generate_seed()))
    return g

def parse_edgelist(path: str, weights: str) -> Tuple["nx.DiGraph", int, int]:
    g = nx.DiGraph()
    has_weight = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                u, v = map(int, parts)
                w = 1.0
            elif len(parts) >= 3:
                u, v = map(int, parts[:2])
                w = float(parts[2])
                has_weight = True
            else:
                continue
            g.add_edge(u, v, weight=w)
    if not has_weight:
        assign_weights(g, weights, random.Random(get_or_generate_seed()))
    n = g.number_of_nodes()
    m = g.number_of_edges()
    return g, n, m

def assign_weights(g: "nx.DiGraph", weights: str, rng: random.Random) -> None:
    if weights == "unit":
        for u, v in g.edges():
            g[u][v]["weight"] = 1.0
        return
    if weights == "lo":
        wfun = lambda: rng.uniform(1.0, 2.0)
    elif weights == "mid":
        wfun = lambda: rng.uniform(1.0, 10.0)
    elif weights == "hi":
        wfun = lambda: rng.uniform(0.5, 100.0)
    else:
        wfun = lambda: rng.uniform(1e-3, 1e3)
    for u, v in g.edges():
        g[u][v]["weight"] = float(wfun())

# ------------- Algorithms -------------

def run_dijkstra(g: "nx.DiGraph", s: int) -> Dict[int, float]:
    return nx.single_source_dijkstra_path_length(g, source=s, weight="weight")

def run_bellman_ford(g: "nx.DiGraph", s: int) -> Dict[int, float]:
    return nx.single_source_bellman_ford_path_length(g, source=s, weight="weight")

def _make_compact_index(g: "nx.DiGraph") -> Tuple[Dict[int, int], List[int]]:
    """Map arbitrary node ids to [0..n-1] and return mapping and inverse list."""
    nodes = list(g.nodes())
    id2idx = {u: i for i, u in enumerate(nodes)}
    idx2id = nodes  # idx -> original id
    return id2idx, idx2id

def run_external(cmd: str, g: "nx.DiGraph", s: int) -> Dict[int, float]:
    """
    Stream graph to external solver using compact [0..n-1] ids,
    then remap solver output back to original ids.
    Accepts two stdout formats:
      - One line with n whitespace separated distances
      - n lines with "node distance"
    Missing nodes default to inf.
    """
    n = g.number_of_nodes()
    m = g.number_of_edges()

    id2idx, idx2id = _make_compact_index(g)
    s_idx = id2idx.get(s, None)
    if s_idx is None:
        raise ValueError("Source not in graph")

    # launch process
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line buffered
    )

    try:
        # header
        proc.stdin.write(f"{n} {m} {s_idx}\n")
        # stream edges
        for u, v, data in g.edges(data=True):
            w = data.get("weight", 1.0)
            proc.stdin.write(f"{id2idx[u]} {id2idx[v]} {w}\n")
        proc.stdin.close()
        stdout, stderr = proc.communicate()
    except Exception:
        proc.kill()
        raise

    if proc.returncode != 0:
        msg = (stderr or "")[:500]
        raise RuntimeError(f"External command failed with code {proc.returncode}. stderr: {msg}")

    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("External command produced no output")

    tokens = text.split()
    dists_idx: Dict[int, float] = {}
    if len(tokens) == n:
        # order corresponds to 0..n-1
        for i, tok in enumerate(tokens):
            try:
                dists_idx[i] = float(tok)
            except ValueError:
                dists_idx[i] = math.inf
    else:
        # parse per-line "node dist"
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    uu = int(parts[0])
                    dd = float(parts[1])
                except ValueError:
                    continue
                if 0 <= uu < n:
                    dists_idx[uu] = dd

    # fill missing as inf
    if len(dists_idx) != n:
        for i in range(n):
            if i not in dists_idx:
                dists_idx[i] = math.inf

    # remap back to original ids
    dists: Dict[int, float] = {}
    for i, d in dists_idx.items():
        u = idx2id[i]
        dists[u] = d
    return dists

def run_dmsy(g: "nx.DiGraph", s: int) -> Dict[int, float]:
    """
    Placeholder experimental algorithm, not the true DMSY complexity.
    Kept for experimentation. Do not rely on its asymptotics.
    """
    from collections import deque
    dist = {node: float("inf") for node in g.nodes()}
    dist[s] = 0.0
    # lightweight SPFA style with small queue heuristic
    inq = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        inq.discard(u)
        du = dist[u]
        for v in g.successors(u):
            w = g[u][v]["weight"]
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                if v not in inq:
                    q.append(v)
                    inq.add(v)
    return dist

ALGO_FUN = {
    "dijkstra": run_dijkstra,
    "bellman_ford": run_bellman_ford,
    "dmsy": run_dmsy,
}

# ------------- Benchmark runner -------------
# helper function, tested only on linux
def _current_rss_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    # fallback best effort
    if resource is not None:
        try:
            val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if val > 10**7:  #bytes ?
                return val / (1024 * 1024)
            else:           #KiB ?
                return val / 1024
        except Exception:
            pass
    return float("nan")

def measure(func, *args, **kwargs):
    tracemalloc.start()
    rss_before = _current_rss_mb()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _current_rss_mb()
    # peak_py is bytes
    python_mb = peak_py / (1024 * 1024)
    # report total working set
    rss_mb = rss_after if not math.isnan(rss_after) else float("nan")
    return res, t1 - t0, python_mb, rss_mb

def verify(reference: Dict[int, float], other: Dict[int, float], atol=1e-7, rtol=1e-7) -> Tuple[bool, int]:
    bad = 0
    # union of keys to be safe
    keys = set(reference.keys()) | set(other.keys())
    for k in keys:
        v = reference.get(k, math.inf)
        ov = other.get(k, math.inf)
        if math.isinf(v) and math.isinf(ov):
            continue
        if not math.isclose(v, ov, rel_tol=rtol, abs_tol=atol):
            bad += 1
    return bad == 0, bad

def build_graph_from_args(args: argparse.Namespace):
    """Build graph based on command line arguments. Returns (g, n, m, gtype)."""
    if args.input:
        g, n, m = parse_edgelist(args.input, args.weights)
        return g, n, m, f"input:{os.path.basename(args.input)}"

    if args.graph == "erdos":
        g = build_erdos(args.n, args.p, args.seed, args.weights)
    elif args.graph == "barabasi":
        g = build_barabasi(args.n, args.m_attach, args.seed, args.weights)
    elif args.graph == "grid":
        g = build_grid(args.rows, args.cols, args.weights)
    elif args.graph == "geometric":
        g = build_geometric(args.n, args.radius, args.seed, args.weights)
    elif args.graph == "livejournal":
        g = build_livejournal(args.weights)
    elif args.graph == "wiki":
        g = build_wiki_talk(args.weights)
    else:
        raise ValueError(f"Unknown graph type: {args.graph}")

    return g, g.number_of_nodes(), g.number_of_edges(), args.graph

def main():
    parser = argparse.ArgumentParser(description="SSSP Benchmark Harness")
    ggrp = parser.add_argument_group("Graph")
    parser.add_argument("--graph", type=str, default="erdos",
                        choices=["erdos", "barabasi", "grid", "geometric", "livejournal", "wiki"],
                        help="Type of graph to generate.")
    parser.add_argument("--input", type=str, default=None, help="Path to edgelist file. Overrides --graph")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--p", type=float, default=5e-5, help="Erdos edge probability")
    parser.add_argument("--m_attach", type=int, default=3, help="Barabasi number of edges to attach")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--cols", type=int, default=300)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--weights", choices=["unit", "lo", "mid", "hi", "custom"], default="mid")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    b = parser.add_argument_group("Benchmark")
    b.add_argument("--algos", type=str, default="dijkstra",
                   help="Comma separated. dmsy,dijkstra,bellman_ford,external")
    b.add_argument("--external-cmd", type=str, default=None, help="Command to run external algorithm")
    b.add_argument("--source", type=int, default=None, help="Source node. Default 0")
    b.add_argument("--trials", type=int, default=3)
    b.add_argument("--out", type=str, default="bench_results.csv")
    b.add_argument("--no_check", action="store_true", help="Skip correctness verification")
    b.add_argument("--atol", type=float, default=1e-6)
    b.add_argument("--rtol", type=float, default=1e-6)

    args = parser.parse_args()

    g, n, m, gtype = build_graph_from_args(args)
    s = 0 if args.source is None else args.source
    if s not in g.nodes:
        raise ValueError(f"Source {s} not in graph")

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a == "external" and not args.external_cmd:
            parser.error("external listed in --algos but --external-cmd not provided")
        if a not in ALGO_FUN and a != "external":
            parser.error(f"Unknown algo {a}")

    # Reference
    ref = None
    if not args.no_check:
        ref, _, _, _ = measure(run_dijkstra, g, s)

    header = ["run_id","algo","graph","n","m","weights","seed","trial","source",
              "time_s","python_mb","rss_mb","ok","mismatches"]
    need_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        run_id = int(time.time())
        for trial in range(1, args.trials + 1):
            for algo in algos:
                if algo == "external":
                    func = lambda gg, ss: run_external(args.external_cmd, gg, ss)
                else:
                    func = ALGO_FUN[algo]
                try:
                    res, t, py_mb, rss_mb = measure(func, g, s)
                    ok = ""
                    mismatches = ""
                    if ref is not None and not args.no_check:
                        okflag, bad = verify(ref, res, args.atol, args.rtol)
                        ok = "yes" if okflag else "no"
                        mismatches = str(bad)
                except Exception as e:
                    t = float("nan")
                    py_mb = float("nan")
                    rss_mb = float("nan")
                    ok = "error"
                    msg = str(e).splitlines()[0]
                    mismatches = (msg[:160] if msg else "error")
                writer.writerow([run_id, algo, gtype, n, m, args.weights, args.seed,
                                 trial, s, f"{t:.6f}", f"{py_mb:.3f}", f"{rss_mb:.3f}", ok, mismatches])
    print(f"Wrote results to {args.out}")

if __name__ == "__main__":
    main()
