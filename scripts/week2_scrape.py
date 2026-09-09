"""
Week 2 — Scrape rapper Wikipedia pages, build link network,
compute real vs degree-preserving null model metrics, export JSON.
"""

import requests
import time
import json
import networkx as nx
import numpy as np
from collections import Counter
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────
API_URL = "https://en.wikipedia.org/w/api.php"
CATEGORIES = [
    "Category:21st-century American rappers",
    "Category:American male rappers",
    "Category:American women rappers",
]
MAX_NODES = 400
MAX_DEPTH = 2  # recurse into subcategories
N_SHUFFLES = 20
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "DTU02805NetworkLab/1.0 (student project; social-graphs-dtu@example.com)"
})


# ── STEP 1: FETCH RAPPER PAGES FROM CATEGORIES (RECURSIVE) ──
def get_category_members_recursive(category, max_pages, depth=0):
    """Fetch page titles from a Wikipedia category, recursing into subcategories."""
    pages = []
    subcats = []

    # Fetch pages
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": category, "cmlimit": 500, "cmtype": "page",
        "format": "json"
    }
    while True:
        r = SESSION.get(API_URL, params=params).json()
        batch = r.get("query", {}).get("categorymembers", [])
        pages.extend(batch)
        if "continue" not in r:
            break
        params["cmcontinue"] = r["continue"]["cmcontinue"]
        time.sleep(0.1)

    # Fetch subcategories
    if depth < MAX_DEPTH:
        params2 = {
            "action": "query", "list": "categorymembers",
            "cmtitle": category, "cmlimit": 500, "cmtype": "subcat",
            "format": "json"
        }
        r2 = SESSION.get(API_URL, params=params2).json()
        subcats = [m["title"] for m in r2.get("query", {}).get("categorymembers", [])]
        time.sleep(0.1)

    return pages, subcats


def collect_all_rappers(categories, max_pages=MAX_NODES):
    """Collect rapper pages from multiple categories with recursion."""
    seen_titles = set()
    all_members = []
    queue = [(cat, 0) for cat in categories]

    while queue and len(all_members) < max_pages:
        cat, depth = queue.pop(0)
        if cat in seen_titles:
            continue
        seen_titles.add(cat)

        pages, subcats = get_category_members_recursive(cat, max_pages, depth)
        for p in pages:
            if p["title"] not in seen_titles and len(all_members) < max_pages:
                # Skip "List of..." pages
                if p["title"].startswith("List of"):
                    continue
                seen_titles.add(p["title"])
                all_members.append(p)

        for sc in subcats:
            if sc not in seen_titles:
                queue.append((sc, depth + 1))

        print(f"  {cat}: {len(pages)} pages, {len(subcats)} subcats (total so far: {len(all_members)})")

    return all_members[:max_pages]


# ── STEP 2: GET OUTGOING WIKI-LINKS FOR EACH PAGE ──────
def get_page_links(title):
    """Return list of internal link targets from a page."""
    all_links = []
    params = {
        "action": "parse", "page": title,
        "prop": "links", "format": "json"
    }
    r = SESSION.get(API_URL, params=params).json()
    if "parse" not in r:
        return []
    for link in r["parse"]["links"]:
        if link.get("ns") == 0 and "exists" in link:
            all_links.append(link["*"].replace(" ", "_"))
    return all_links


# ── STEP 3: BUILD NETWORK ──────────────────────────────
def build_network(members):
    """Build directed graph: edge A→B if A's wiki page links to B."""
    titles = {m["title"].replace(" ", "_") for m in members}
    title_to_name = {m["title"].replace(" ", "_"): m["title"] for m in members}

    G = nx.DiGraph()
    for t in titles:
        G.add_node(t, name=title_to_name[t])

    print(f"Fetching links for {len(titles)} pages...")
    for i, title in enumerate(sorted(titles)):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(titles)}")
        try:
            links = get_page_links(title)
            for target in links:
                if target in titles and target != title:
                    G.add_edge(title, target)
        except Exception as e:
            print(f"  Error on {title}: {e}")
        time.sleep(0.1)

    # Remove isolates (no in or out edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"Removed {len(isolates)} isolates. Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── STEP 4: COMPUTE METRICS ────────────────────────────
def compute_metrics(G_und):
    """Compute clustering, triangles, assortativity, avg path, friendship paradox."""
    if G_und.number_of_nodes() == 0:
        return {"clustering": 0, "triangles": 0, "assortativity": 0,
                "avg_path": 0, "n_communities": 0, "modularity": 0, "friendship_paradox": 0}
    metrics = {}

    # Clustering coefficient (global transitivity)
    metrics["clustering"] = round(nx.transitivity(G_und), 4)

    # Triangle count
    tri_dict = nx.triangles(G_und)
    metrics["triangles"] = sum(tri_dict.values()) // 3

    # Degree assortativity
    try:
        metrics["assortativity"] = round(nx.degree_assortativity_coefficient(G_und), 4)
    except Exception:
        metrics["assortativity"] = 0.0

    # Average shortest path length (giant component only)
    if nx.is_connected(G_und):
        metrics["avg_path"] = round(nx.average_shortest_path_length(G_und), 4)
    else:
        gcc_nodes = max(nx.connected_components(G_und), key=len)
        gcc = G_und.subgraph(gcc_nodes)
        metrics["avg_path"] = round(nx.average_shortest_path_length(gcc), 4)
        metrics["gcc_size"] = len(gcc_nodes)

    # Communities (greedy modularity)
    comms = list(nx.community.greedy_modularity_communities(G_und))
    metrics["n_communities"] = len(comms)
    metrics["modularity"] = round(nx.community.modularity(G_und, comms), 4)

    # Friendship paradox: fraction of nodes whose neighbors have higher avg degree
    degrees = dict(G_und.degree())
    paradox_count = 0
    for node in G_und.nodes():
        neighbors = list(G_und.neighbors(node))
        if not neighbors:
            continue
        avg_neighbor_deg = np.mean([degrees[n] for n in neighbors])
        if avg_neighbor_deg > degrees[node]:
            paradox_count += 1
    metrics["friendship_paradox"] = round(paradox_count / G_und.number_of_nodes(), 4)

    return metrics


# ── STEP 5: NULL MODEL (DEGREE-PRESERVING SHUFFLE) ─────
def make_null_model(G_und, n_swaps=None):
    """Create a degree-preserving random graph via double edge swaps."""
    H = G_und.copy()
    if n_swaps is None:
        n_swaps = H.number_of_edges() * 10
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 20)
    except nx.NetworkXAlgorithmError:
        pass  # couldn't complete all swaps, partial is fine
    return H


# ── STEP 6: COMMUNITY DETECTION FOR VIZ ────────────────
def assign_communities(G_und):
    """Assign community IDs to nodes."""
    comms = list(nx.community.greedy_modularity_communities(G_und))
    node_to_comm = {}
    for i, comm in enumerate(comms):
        for node in comm:
            node_to_comm[node] = i
    return node_to_comm, len(comms)


# ── STEP 7: EXPORT JSON ────────────────────────────────
def export_json(G, G_und, real_metrics, null_runs, output_dir):
    """Export network.json and charts.json for the Week 2 article."""
    output_dir.mkdir(parents=True, exist_ok=True)

    node_to_comm, n_comms = assign_communities(G_und)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # ── rappers_network.json ──
    nodes = []
    for node in G.nodes():
        nodes.append({
            "id": node,
            "name": G.nodes[node].get("name", node.replace("_", " ")),
            "in_deg": in_deg.get(node, 0),
            "out_deg": out_deg.get(node, 0),
            "community": node_to_comm.get(node, 0)
        })

    links = [{"source": u, "target": v} for u, v in G.edges()]

    network_data = {
        "nodes": nodes,
        "links": links,
        "num_communities": n_comms
    }

    with open(output_dir / "rappers_network.json", "w") as f:
        json.dump(network_data, f)
    print(f"Wrote rappers_network.json ({len(nodes)} nodes, {len(links)} links)")

    # ── rappers_charts.json ──
    # Null model stats
    null_keys = list(null_runs[0].keys())
    null_mean = {k: round(float(np.mean([r[k] for r in null_runs])), 4) for k in null_keys}
    null_std = {k: round(float(np.std([r[k] for r in null_runs])), 4) for k in null_keys}

    # Degree distribution
    in_counts = Counter(in_deg.values())
    out_counts = Counter(out_deg.values())

    # Top nodes by in-degree
    top_in = sorted(nodes, key=lambda x: x["in_deg"], reverse=True)[:15]

    charts_data = {
        "comparison": {
            "real": real_metrics,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_runs": null_runs
        },
        "degree_dist": {
            "in": [{"deg": k, "count": v} for k, v in sorted(in_counts.items())],
            "out": [{"deg": k, "count": v} for k, v in sorted(out_counts.items())]
        },
        "top_nodes": [{"name": n["name"], "in_deg": n["in_deg"], "out_deg": n["out_deg"]} for n in top_in],
        "network_stats": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_undirected_edges": G_und.number_of_edges(),
            "density": round(nx.density(G_und), 4),
            "avg_degree": round(2 * G_und.number_of_edges() / G_und.number_of_nodes(), 2)
        }
    }

    with open(output_dir / "rappers_charts.json", "w") as f:
        json.dump(charts_data, f)
    print(f"Wrote rappers_charts.json")


# ── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Step 1: Fetching category members ===")
    members = collect_all_rappers(CATEGORIES, MAX_NODES)
    print(f"Found {len(members)} rapper pages total")

    print("\n=== Step 2-3: Building network ===")
    G = build_network(members)

    print("\n=== Step 4: Computing real metrics ===")
    G_und = G.to_undirected()
    real_metrics = compute_metrics(G_und)
    print("Real metrics:", json.dumps(real_metrics, indent=2))

    print(f"\n=== Step 5: Running {N_SHUFFLES} null model shuffles ===")
    null_runs = []
    for i in range(N_SHUFFLES):
        H = make_null_model(G_und)
        m = compute_metrics(H)
        null_runs.append(m)
        if (i + 1) % 5 == 0:
            print(f"  Shuffle {i+1}/{N_SHUFFLES} done")

    print("\n=== Step 6-7: Exporting JSON ===")
    export_json(G, G_und, real_metrics, null_runs, OUTPUT_DIR)

    print("\n=== Done! ===")
    print(f"Null model mean clustering: {np.mean([r['clustering'] for r in null_runs]):.4f}")
    print(f"Real clustering:            {real_metrics['clustering']:.4f}")
    ratio = real_metrics['clustering'] / max(np.mean([r['clustering'] for r in null_runs]), 1e-6)
    print(f"Ratio (real/null):          {ratio:.1f}x")
