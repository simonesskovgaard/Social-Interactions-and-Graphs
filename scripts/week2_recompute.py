"""
Re-compute new metrics from cached network data without re-scraping Wikipedia.
Loads rappers_network.json, rebuilds the graph, runs null model + new computations,
and writes updated rappers_charts.json.
"""
import json
import networkx as nx
import numpy as np
from pathlib import Path

# Import functions from the main script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from week2_scrape import (
    compute_metrics, make_null_model, assign_communities,
    compute_z_scores_and_pvalues, compute_ccdf, compute_ba_ccdf,
    compute_local_clustering_histogram, compute_ws_sweep, N_SHUFFLES
)
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"

def main():
    # Load cached network
    with open(DATA_DIR / "rappers_network.json") as f:
        net = json.load(f)

    print(f"Loaded network: {len(net['nodes'])} nodes, {len(net['links'])} links")

    # Rebuild directed graph
    G = nx.DiGraph()
    for node in net["nodes"]:
        G.add_node(node["id"], name=node["name"])
    for link in net["links"]:
        G.add_edge(link["source"], link["target"])

    G_und = G.to_undirected()
    N = G.number_of_nodes()

    print(f"Directed: {G.number_of_edges()} edges, Undirected: {G_und.number_of_edges()} edges")

    # Real metrics
    print("\nComputing real metrics...")
    real_metrics = compute_metrics(G_und)
    print(f"Real metrics: {json.dumps(real_metrics, indent=2)}")

    # Null model runs
    print(f"\nRunning {N_SHUFFLES} null model shuffles...")
    null_runs = []
    for i in range(N_SHUFFLES):
        H = make_null_model(G_und)
        m = compute_metrics(H)
        null_runs.append(m)
        if (i + 1) % 5 == 0:
            print(f"  Shuffle {i+1}/{N_SHUFFLES} done")

    # Z-scores and p-values
    null_keys = list(null_runs[0].keys())
    null_mean = {k: round(float(np.mean([r[k] for r in null_runs])), 4) for k in null_keys}
    null_std = {k: round(float(np.std([r[k] for r in null_runs])), 4) for k in null_keys}
    z_scores, p_values = compute_z_scores_and_pvalues(real_metrics, null_runs)
    print(f"\nZ-scores: {json.dumps(z_scores, indent=2)}")
    print(f"P-values: {json.dumps(p_values, indent=2)}")

    # Degree distributions
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_counts = Counter(in_deg.values())
    out_counts = Counter(out_deg.values())

    # CCDF
    avg_deg = 2 * G_und.number_of_edges() / N
    in_ccdf = compute_ccdf(in_counts, N)
    ba_ccdf = compute_ba_ccdf(N, avg_deg)
    print(f"\nCCDF: {len(in_ccdf)} rapper points, {len(ba_ccdf)} BA points")

    # Local clustering
    local_clust_hist, mean_local, _ = compute_local_clustering_histogram(G_und)
    print(f"Local clustering: mean={mean_local}, transitivity={real_metrics['clustering']}")

    # WS sweep
    print("\nComputing Watts-Strogatz sweep...")
    ws_sweep, ws_C0, ws_L0 = compute_ws_sweep(N, avg_deg)
    rapper_norm_C = round(real_metrics["clustering"] / ws_C0, 4) if ws_C0 > 0 else 0
    rapper_norm_L = round(real_metrics["avg_path"] / ws_L0, 4) if ws_L0 > 0 else 0
    print(f"Rapper: norm_C={rapper_norm_C}, norm_L={rapper_norm_L}")

    # Communities + top nodes
    node_to_comm, n_comms = assign_communities(G_und)
    nodes = []
    for node in G.nodes():
        nodes.append({
            "id": node,
            "name": G.nodes[node].get("name", node.replace("_", " ")),
            "in_deg": in_deg.get(node, 0),
            "out_deg": out_deg.get(node, 0),
            "community": node_to_comm.get(node, 0)
        })
    top_in = sorted(nodes, key=lambda x: x["in_deg"], reverse=True)[:15]

    # Build charts data
    charts_data = {
        "comparison": {
            "real": real_metrics,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_runs": null_runs,
            "z_scores": z_scores,
            "p_values": p_values,
        },
        "degree_dist": {
            "in": [{"deg": k, "count": v} for k, v in sorted(in_counts.items())],
            "out": [{"deg": k, "count": v} for k, v in sorted(out_counts.items())]
        },
        "ccdf": {
            "rapper": in_ccdf,
            "ba_model": ba_ccdf,
        },
        "local_clustering": {
            "histogram": local_clust_hist,
            "mean_local": mean_local,
            "transitivity": real_metrics["clustering"],
        },
        "ws_sweep": {
            "data": ws_sweep,
            "rapper_norm_clustering": rapper_norm_C,
            "rapper_norm_path": rapper_norm_L,
            "ws_C0": round(ws_C0, 4),
            "ws_L0": round(ws_L0, 4),
        },
        "top_nodes": [{"name": n["name"], "in_deg": n["in_deg"], "out_deg": n["out_deg"]} for n in top_in],
        "network_stats": {
            "n_nodes": N,
            "n_edges": G.number_of_edges(),
            "n_undirected_edges": G_und.number_of_edges(),
            "density": round(nx.density(G_und), 4),
            "avg_degree": round(avg_deg, 2)
        }
    }

    with open(DATA_DIR / "rappers_charts.json", "w") as f:
        json.dump(charts_data, f)
    print(f"\nWrote rappers_charts.json")
    print("Done!")

if __name__ == "__main__":
    main()
