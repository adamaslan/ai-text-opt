#!/usr/bin/env python3
"""
Knowledge Graph – 3 distinct visualisations.

Reads all ChromaDB collections, computes cosine-similarity edges,
then renders three very different views:

  1. galaxy.png        – dark radial spring layout, glowing nodes, arc edges
  2. heatmap.png       – similarity matrix heatmap sorted by collection
  3. chord.png         – chord diagram showing cross-collection flow

Requirements:
    pip install chromadb networkx matplotlib numpy scipy

Usage:
    python visualize_graph.py
    python visualize_graph.py --threshold 0.65 --sample 60
    python visualize_graph.py --out-dir ./visuals
"""

from __future__ import annotations

import argparse
import math
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = str(Path(__file__).parent / "chromadb_storage")
DEFAULT_THRESHOLD = 0.72
DEFAULT_SAMPLE = 80

# Tags that will be rendered as shared keyword hub nodes in the idea-network view
KEYWORD_TAGS = [
    "beauty", "expression", "authenticity", "silence", "nietzsche",
    "deleuze", "schopenhauer", "enlightenment", "eudaimonia", "camus",
    "buddhism", "meditation", "risk", "freedom", "meaning", "ritual",
    "society", "vulnerability", "cosmos", "creation", "ethics",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load graph data from ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _cosine(a, b, na=None, nb=None):
    na = na or _norm(a)
    nb = nb or _norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def load_graph(
    db_path: str = CHROMA_DB_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    sample: int = DEFAULT_SAMPLE,
) -> Tuple[nx.Graph, Dict[str, str], Dict[str, str]]:
    """
    Returns:
        G               – networkx Graph with 'collection', 'title', 'tags' node attrs
        node_collection – {node_id: collection_name}
        node_label      – {node_id: short display label}
    """
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No ChromaDB collections found at {db_path}")

    G = nx.Graph()
    node_collection: Dict[str, str] = {}
    node_label: Dict[str, str] = {}
    vectors: Dict[str, List[float]] = {}
    norms: Dict[str, float] = {}

    for coll_obj in collections:
        name = coll_obj.name
        coll = client.get_collection(name)
        data = coll.get(include=["embeddings", "metadatas"], limit=sample)
        ids = data.get("ids") or []
        embs = data.get("embeddings")
        if embs is None:
            embs = []
        metas = data.get("metadatas") or []

        for doc_id, vec, meta in zip(ids, embs, metas):
            meta = meta or {}
            v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            n = _norm(v)
            if not v or n < 1e-6:
                continue
            nid = f"{name}::{doc_id}"
            title = meta.get("title", doc_id)
            # Shorten label: first 4 words
            short = " ".join(title.split()[:4])
            G.add_node(nid, collection=name, title=title, tags=meta.get("tags", ""))
            node_collection[nid] = name
            node_label[nid] = short
            vectors[nid] = v
            norms[nid] = n

    logger.info("Nodes loaded: %d", G.number_of_nodes())

    # Build edges
    node_list = list(G.nodes())
    edge_count = 0
    for i in range(len(node_list)):
        a = node_list[i]
        for j in range(i + 1, len(node_list)):
            b = node_list[j]
            sim = _cosine(vectors[a], vectors[b], norms[a], norms[b])
            if sim >= threshold:
                G.add_edge(a, b, weight=sim,
                           cross=(node_collection[a] != node_collection[b]))
                edge_count += 1

    logger.info("Edges (threshold=%.2f): %d", threshold, edge_count)
    return G, node_collection, node_label


def collection_palette(collections: List[str]) -> Dict[str, str]:
    """Assign a visually distinct hex colour to each collection."""
    base_colors = [
        "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
        "#C3A6FF", "#FF9F43", "#54A0FF", "#5F27CD",
        "#00D2D3", "#FF9FF3",
    ]
    return {c: base_colors[i % len(base_colors)] for i, c in enumerate(sorted(collections))}


# ─────────────────────────────────────────────────────────────────────────────
# Visual 1 – Galaxy: exploded spring layout, all nodes labeled, keyword halos
# ─────────────────────────────────────────────────────────────────────────────

def draw_galaxy(
    G: nx.Graph,
    node_collection: Dict[str, str],
    node_label: Dict[str, str],
    palette: Dict[str, str],
    out_path: str = "galaxy.png",
) -> None:
    """
    Dark spring layout with:
    - Strong repulsion so the dense center explodes outward
    - Every node labeled (small font) with high-degree nodes highlighted
    - Cross-collection edges in gold, same-collection edges dim
    - Keyword tag halos: translucent coloured rings around nodes that share a tag
    """
    logger.info("Rendering galaxy…")

    fig, ax = plt.subplots(figsize=(26, 24), facecolor="#080818")
    ax.set_facecolor("#080818")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes loaded", color="white",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_nodes = G.number_of_nodes()

    # ── Layout: large k separates nodes strongly ─────────────────────────────
    # Use kamada_kawai for better separation when graph is connected,
    # fallback to spring for large/sparse graphs
    if n_nodes <= 150:
        try:
            pos = nx.kamada_kawai_layout(G, weight="weight")
        except Exception:
            pos = nx.spring_layout(G, k=3.5 / math.sqrt(n_nodes), seed=7, iterations=120)
    else:
        pos = nx.spring_layout(G, k=3.5 / math.sqrt(n_nodes), seed=7, iterations=120)

    degrees = dict(G.degree())
    max_deg = max(degrees.values(), default=1)

    # ── Keyword tag halos: draw translucent circles at tag-shared positions ───
    # For each keyword tag, find all nodes that mention it, draw a soft halo
    tag_colours = [
        "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF", "#C3A6FF",
        "#FF9F43", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9FF3",
        "#FFEAA7", "#DFE6E9", "#B2BEC3", "#6C5CE7", "#E17055",
        "#00B894", "#FDCB6E", "#E84393", "#2D3436", "#74B9FF",
        "#FD79A8",
    ]
    tag_palette = {tag: tag_colours[i % len(tag_colours)]
                   for i, tag in enumerate(KEYWORD_TAGS)}

    for tag in KEYWORD_TAGS:
        tagged_nodes = [
            nid for nid in G.nodes()
            if tag in G.nodes[nid].get("tags", "").lower()
        ]
        if len(tagged_nodes) < 2:
            continue
        tc = tag_palette[tag]
        for nid in tagged_nodes:
            x, y = pos[nid]
            circle = plt.Circle((x, y), 0.025, color=tc, alpha=0.13,
                                 linewidth=0, zorder=2)
            ax.add_patch(circle)

    # ── Draw edges ────────────────────────────────────────────────────────────
    cross_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("cross")]
    same_edges  = [(u, v, d) for u, v, d in G.edges(data=True) if not d.get("cross")]

    # Same-collection: weight-scaled opacity, thin
    for u, v, d in same_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        alpha = 0.05 + 0.15 * (d.get("weight", 0.65) - 0.65) / 0.35
        ax.plot([x0, x1], [y0, y1], color="#FFFFFF",
                alpha=alpha, linewidth=0.5, zorder=1)

    # Cross-collection: gold, opacity by weight
    for u, v, d in cross_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        alpha = 0.2 + 0.5 * (d.get("weight", 0.65) - 0.65) / 0.35
        lw = 0.8 + 1.5 * (d.get("weight", 0.65) - 0.65) / 0.35
        ax.plot([x0, x1], [y0, y1], color="#FFD700",
                alpha=alpha, linewidth=lw, zorder=2)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for nid in G.nodes():
        coll = node_collection[nid]
        colour = palette.get(coll, "#AAAAAA")
        deg = degrees[nid]
        base_size = 30 + 220 * (deg / max_deg)

        x, y = pos[nid]
        ax.scatter(x, y, s=base_size * 5,  color=colour, alpha=0.05, zorder=3)
        ax.scatter(x, y, s=base_size * 2,  color=colour, alpha=0.20, zorder=4)
        ax.scatter(x, y, s=base_size,       color=colour, alpha=0.92,
                   edgecolors="white", linewidths=0.3, zorder=5)

    # ── Labels: every node, size by degree ───────────────────────────────────
    for nid in G.nodes():
        deg = degrees[nid]
        x, y = pos[nid]
        label = node_label.get(nid, nid.split("::")[-1])
        # Scale font: 5.5 for isolated nodes up to 9 for hubs
        fsize = 5.5 + 4.0 * (deg / max_deg)
        weight = "bold" if deg >= max_deg * 0.4 else "normal"
        ax.text(
            x, y + 0.022, label,
            fontsize=fsize, color="white", ha="center", va="bottom",
            fontweight=weight, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#080818")],
        )

    # ── Keyword tag legend (top-right, compact) ───────────────────────────────
    present_tags = [
        tag for tag in KEYWORD_TAGS
        if sum(1 for nid in G.nodes() if tag in G.nodes[nid].get("tags", "").lower()) >= 2
    ]
    tag_handles = [
        mpatches.Patch(color=tag_palette[t], label=t, alpha=0.7)
        for t in present_tags[:14]   # cap at 14 to avoid overflow
    ]
    if tag_handles:
        leg_tags = ax.legend(
            handles=tag_handles, title="Keyword Tags", title_fontsize=8,
            loc="upper right", fontsize=7, ncol=2,
            framealpha=0.35, facecolor="#080818", edgecolor="#555555",
            labelcolor="white",
        )
        leg_tags.get_title().set_color("white")
        ax.add_artist(leg_tags)

    # ── Collection legend (bottom-left) ──────────────────────────────────────
    coll_handles = [
        mpatches.Patch(color=palette[c], label=c)
        for c in sorted(palette)
        if any(node_collection[n] == c for n in G.nodes())
    ]
    ax.legend(
        handles=coll_handles, title="Collections", title_fontsize=8,
        loc="lower left", fontsize=8,
        framealpha=0.35, facecolor="#080818", edgecolor="#555555",
        labelcolor="white",
    ).get_title().set_color("white")

    ax.set_title(
        "Knowledge Graph — Galaxy View  (gold = cross-collection · halos = shared keyword)",
        color="white", fontsize=15, pad=16,
        path_effects=[pe.withStroke(linewidth=3, foreground="#080818")],
    )
    ax.text(
        0.99, 0.01,
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges",
        transform=ax.transAxes, fontsize=7.5,
        color="#777777", ha="right", va="bottom",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#080818")
    plt.close(fig)
    logger.info("✓ Galaxy saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visual 2 – Idea Network: bipartite-style map of ideas + keyword hub nodes
# ─────────────────────────────────────────────────────────────────────────────

def draw_heatmap(
    G: nx.Graph,
    node_collection: Dict[str, str],
    node_label: Dict[str, str],
    palette: Dict[str, str],
    out_path: str = "heatmap.png",
    max_idea_nodes: int = 70,
) -> None:
    """
    Idea-connection network:
    - Idea nodes (coloured by collection) laid out by spring
    - Keyword hub nodes (white diamonds) drawn as shared anchors
    - Edges between idea nodes: cosine-similarity lines, width/alpha by weight
    - Edges from ideas to keyword hubs: thin dotted lines
    - Every idea node is labeled; keyword hubs get a bold tag label
    """
    logger.info("Rendering idea-network…")

    BG = "#0D0D1F"
    fig, ax = plt.subplots(figsize=(24, 22), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # ── Select idea nodes (most connected if too many) ────────────────────────
    all_idea_nodes = list(G.nodes())
    deg = dict(G.degree())
    if len(all_idea_nodes) > max_idea_nodes:
        all_idea_nodes = sorted(all_idea_nodes, key=lambda n: deg[n], reverse=True)[:max_idea_nodes]

    idea_set = set(all_idea_nodes)

    # Sub-graph of just selected idea nodes
    SG = G.subgraph(all_idea_nodes).copy()

    # ── Identify keyword hubs: tags shared by ≥2 selected nodes ──────────────
    tag_to_nodes: Dict[str, List[str]] = defaultdict(list)
    for nid in all_idea_nodes:
        raw_tags = G.nodes[nid].get("tags", "")
        for tag in [t.strip().lower() for t in raw_tags.split(",") if t.strip()]:
            if tag in KEYWORD_TAGS:
                tag_to_nodes[tag].append(nid)

    active_tags = {tag: nodes for tag, nodes in tag_to_nodes.items() if len(nodes) >= 2}

    # ── Build augmented graph: idea nodes + keyword hub nodes ─────────────────
    AG = nx.Graph()
    for nid in all_idea_nodes:
        AG.add_node(nid, kind="idea")
    for tag in active_tags:
        hub_id = f"__tag__{tag}"
        AG.add_node(hub_id, kind="tag", label=tag)
        for nid in active_tags[tag]:
            AG.add_edge(hub_id, nid, kind="tag_link")

    # Copy idea-idea edges into AG
    for u, v, d in SG.edges(data=True):
        AG.add_edge(u, v, kind="idea_link", weight=d.get("weight", 0.65),
                    cross=d.get("cross", False))

    # ── Layout: spring on full augmented graph ────────────────────────────────
    n_total = AG.number_of_nodes()
    pos = nx.spring_layout(AG, k=3.0 / math.sqrt(max(n_total, 1)),
                           seed=42, iterations=150, weight=None)

    # ── Draw idea-idea edges ──────────────────────────────────────────────────
    for u, v, d in AG.edges(data=True):
        if d.get("kind") != "idea_link":
            continue
        w = d.get("weight", 0.65)
        cross = d.get("cross", False)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "#FFD700" if cross else "#AAAAAA"
        alpha = 0.12 + 0.5 * (w - 0.65) / 0.35
        lw = 0.4 + 2.5 * (w - 0.65) / 0.35
        ax.plot([x0, x1], [y0, y1], color=color,
                alpha=min(alpha, 0.75), linewidth=lw, zorder=1)

    # ── Draw tag-link edges (dotted, per-tag colour) ──────────────────────────
    tag_colours = [
        "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF", "#C3A6FF",
        "#FF9F43", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9FF3",
        "#FFEAA7", "#E17055", "#00B894", "#FDCB6E", "#74B9FF",
        "#FD79A8", "#B2BEC3", "#6C5CE7", "#E84393", "#DFE6E9",
        "#55EFC4",
    ]
    tag_palette = {tag: tag_colours[i % len(tag_colours)]
                   for i, tag in enumerate(sorted(active_tags.keys()))}

    for u, v, d in AG.edges(data=True):
        if d.get("kind") != "tag_link":
            continue
        # u is hub_id, v is idea node (or vice versa)
        hub = u if u.startswith("__tag__") else v
        idea = v if u.startswith("__tag__") else u
        tag = hub.replace("__tag__", "")
        tc = tag_palette.get(tag, "#FFFFFF")
        x0, y0 = pos[hub]
        x1, y1 = pos[idea]
        ax.plot([x0, x1], [y0, y1], color=tc,
                alpha=0.25, linewidth=0.8, linestyle=":", zorder=2)

    # ── Draw idea nodes ───────────────────────────────────────────────────────
    max_deg_idea = max((deg.get(n, 0) for n in all_idea_nodes), default=1)
    for nid in all_idea_nodes:
        coll = node_collection[nid]
        colour = palette.get(coll, "#AAAAAA")
        d = deg.get(nid, 0)
        size = 40 + 180 * (d / max_deg_idea)
        x, y = pos[nid]
        ax.scatter(x, y, s=size * 3, color=colour, alpha=0.08, zorder=3)
        ax.scatter(x, y, s=size,     color=colour, alpha=0.90,
                   edgecolors="white", linewidths=0.35, zorder=4)

        # Label: full title, word-wrapped at ~20 chars
        label = node_label.get(nid, nid.split("::")[-1])
        fsize = 5.0 + 3.0 * (d / max_deg_idea)
        ax.text(x, y + 0.018, label,
                fontsize=fsize, color="white", ha="center", va="bottom",
                fontweight="bold" if d >= max_deg_idea * 0.35 else "normal",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=BG)])

    # ── Draw keyword hub nodes (diamonds) ─────────────────────────────────────
    for hub_id in AG.nodes():
        if not hub_id.startswith("__tag__"):
            continue
        tag = hub_id.replace("__tag__", "")
        tc = tag_palette.get(tag, "#FFFFFF")
        x, y = pos[hub_id]
        n_linked = len(active_tags[tag])
        size = 120 + 60 * n_linked
        # Diamond marker
        ax.scatter(x, y, s=size * 2.5, color=tc, alpha=0.12,
                   marker="D", zorder=5)
        ax.scatter(x, y, s=size, color=tc, alpha=0.85,
                   marker="D", edgecolors="white", linewidths=0.6, zorder=6)
        ax.text(x, y - 0.030, f"#{tag}",
                fontsize=7.5, color=tc, ha="center", va="top",
                fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])

    # ── Legends ───────────────────────────────────────────────────────────────
    coll_handles = [
        mpatches.Patch(color=palette[c], label=c)
        for c in sorted(palette)
        if any(node_collection[n] == c for n in all_idea_nodes)
    ]
    leg1 = ax.legend(
        handles=coll_handles, title="Collections", title_fontsize=8,
        loc="lower left", fontsize=8,
        framealpha=0.4, facecolor=BG, edgecolor="#444444",
        labelcolor="white",
    )
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)

    tag_handles = [
        mpatches.Patch(color=tag_palette[t], label=f"#{t}")
        for t in sorted(active_tags.keys())
    ]
    if tag_handles:
        leg2 = ax.legend(
            handles=tag_handles, title="Keyword Hubs ◆", title_fontsize=8,
            loc="lower right", fontsize=7.5, ncol=2,
            framealpha=0.4, facecolor=BG, edgecolor="#444444",
            labelcolor="white",
        )
        leg2.get_title().set_color("white")
        ax.add_artist(leg2)

    ax.set_title(
        "Idea Connection Network  (◆ = keyword hub · gold = cross-collection · dotted = keyword link)",
        color="white", fontsize=14, pad=14,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
    )
    ax.text(
        0.99, 0.01,
        f"{len(all_idea_nodes)} idea nodes · {len(active_tags)} keyword hubs",
        transform=ax.transAxes, fontsize=7.5,
        color="#777777", ha="right", va="bottom",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("✓ Idea-network saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visual 3 – Chord diagram: cross-collection edge-weight flow
# ─────────────────────────────────────────────────────────────────────────────

def draw_chord(
    G: nx.Graph,
    node_collection: Dict[str, str],
    palette: Dict[str, str],
    out_path: str = "chord.png",
) -> None:
    """
    Circular chord diagram.
    Each arc segment = one collection (sized by node count).
    Each chord = total cross-collection edge weight between two collections.
    Intra-collection weight shown as a thin self-loop arc on the segment.
    """
    logger.info("Rendering chord diagram…")

    collections = sorted({node_collection[n] for n in G.nodes()})
    if len(collections) < 2:
        logger.warning("Need ≥2 collections for chord diagram.")
        return

    C = len(collections)
    coll_idx = {c: i for i, c in enumerate(collections)}

    # Build C×C weight matrix
    W = np.zeros((C, C), dtype=float)
    for u, v, d in G.edges(data=True):
        i = coll_idx[node_collection[u]]
        j = coll_idx[node_collection[v]]
        W[i, j] += d["weight"]
        W[j, i] += d["weight"]
    # Halve symmetric entries (each edge counted once per direction above)
    for i in range(C):
        for j in range(i + 1, C):
            W[i, j] /= 2
            W[j, i] /= 2

    # Node counts per collection (for segment size)
    node_counts = np.array([
        sum(1 for n in G.nodes() if node_collection[n] == c)
        for c in collections
    ], dtype=float)
    total_nodes = node_counts.sum()

    # Angular positions: each collection occupies an arc proportional to node count
    GAP = 0.04   # radians gap between segments
    total_gap = GAP * C
    arc_sizes = (node_counts / total_nodes) * (2 * math.pi - total_gap)

    starts = np.zeros(C)
    for i in range(1, C):
        starts[i] = starts[i - 1] + arc_sizes[i - 1] + GAP
    mids = starts + arc_sizes / 2

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="#0A0A0F")
    ax.set_facecolor("#0A0A0F")
    ax.set_aspect("equal")
    ax.axis("off")

    R_outer = 1.0    # outer ring radius
    R_inner = 0.88   # inner chord anchor radius
    R_label = 1.10   # label radius

    def arc_path(theta_start, theta_end, r, n_pts=120):
        """Return (x, y) arrays for a circular arc."""
        thetas = np.linspace(theta_start, theta_end, n_pts)
        return r * np.cos(thetas), r * np.sin(thetas)

    # ── Draw outer segments ───────────────────────────────────────────────────
    for i, coll in enumerate(collections):
        colour = palette.get(coll, "#888888")
        t0, t1 = starts[i], starts[i] + arc_sizes[i]

        # Outer arc (thick)
        xs, ys = arc_path(t0, t1, R_outer)
        ax.plot(xs, ys, color=colour, linewidth=12, solid_capstyle="butt",
                alpha=0.85, zorder=4)

        # Inner arc edge (thin, same colour)
        xs2, ys2 = arc_path(t0, t1, R_inner)
        ax.plot(xs2, ys2, color=colour, linewidth=1.5, alpha=0.4, zorder=4)

        # Radial end lines
        for t in [t0, t1]:
            ax.plot([R_inner * math.cos(t), R_outer * math.cos(t)],
                    [R_inner * math.sin(t), R_outer * math.sin(t)],
                    color=colour, linewidth=1.0, alpha=0.5, zorder=4)

        # Label
        mid_t = mids[i]
        lx, ly = R_label * math.cos(mid_t), R_label * math.sin(mid_t)
        ha = "left" if math.cos(mid_t) >= 0 else "right"
        ax.text(lx, ly, coll, fontsize=10, color=colour, ha=ha, va="center",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#0A0A0F")])

        # Node count badge
        bx, by = (R_outer + 0.04) * math.cos(mid_t), (R_outer + 0.04) * math.sin(mid_t)
        ax.text(bx, by, str(int(node_counts[i])),
                fontsize=7, color="white", ha="center", va="center", alpha=0.7)

    # ── Draw chords ───────────────────────────────────────────────────────────
    # Total weight for normalisation (chord opacity)
    max_w = W.max() if W.max() > 0 else 1.0

    for i in range(C):
        for j in range(i, C):
            w = W[i, j]
            if w < 1e-4:
                continue

            alpha = 0.15 + 0.45 * (w / max_w)
            lw = 0.5 + 4.0 * (w / max_w)

            if i == j:
                # Self-loop: small arc on the inner circle
                t0, t1 = starts[i], starts[i] + arc_sizes[i]
                tmid = (t0 + t1) / 2
                span = (t1 - t0) * 0.4
                xs, ys = arc_path(tmid - span, tmid + span, R_inner * 0.93)
                colour = palette.get(collections[i], "#888888")
                ax.plot(xs, ys, color=colour, linewidth=lw * 1.5,
                        alpha=alpha * 0.6, zorder=2)
            else:
                # Bezier chord between two collections
                # Anchor points: middle of each collection's inner arc
                ti = starts[i] + arc_sizes[i] / 2
                tj = starts[j] + arc_sizes[j] / 2
                p0 = np.array([R_inner * math.cos(ti), R_inner * math.sin(ti)])
                p1 = np.array([R_inner * math.cos(tj), R_inner * math.sin(tj)])
                ctrl = np.array([0.0, 0.0])  # control point at origin for inward curve

                # Cubic Bezier via matplotlib Path
                from matplotlib.path import Path as MPath
                from matplotlib.patches import PathPatch

                verts = [p0, ctrl, ctrl, p1]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
                path = MPath(verts, codes)

                # Blend colours of the two collections
                ci = mcolors.to_rgb(palette.get(collections[i], "#888888"))
                cj = mcolors.to_rgb(palette.get(collections[j], "#888888"))
                blend = tuple((a + b) / 2 for a, b in zip(ci, cj))

                patch = PathPatch(path, facecolor="none",
                                  edgecolor=blend, linewidth=lw,
                                  alpha=alpha, zorder=3)
                ax.add_patch(patch)

    # ── Centre annotation ─────────────────────────────────────────────────────
    ax.text(0, 0,
            f"{G.number_of_nodes()}\nnodes",
            ha="center", va="center", fontsize=14,
            color="#CCCCCC", alpha=0.6, zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=palette[c], label=c) for c in collections]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.05, -0.05),
              fontsize=9, framealpha=0.3, facecolor="#0A0A0F",
              edgecolor="white", labelcolor="white")

    ax.set_title(
        "Knowledge Graph — Cross-Collection Chord Diagram",
        color="white", fontsize=15, pad=20,
        path_effects=[pe.withStroke(linewidth=3, foreground="#0A0A0F")],
    )
    ax.text(
        0.99, 0.01,
        "chord width ∝ shared semantic weight",
        transform=ax.transAxes, fontsize=7.5,
        color="#888888", ha="right", va="bottom",
    )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0A0A0F")
    plt.close(fig)
    logger.info("✓ Chord saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 3 knowledge-graph visualisations from ChromaDB"
    )
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity edge threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--sample", "-s", type=int, default=DEFAULT_SAMPLE,
                        help=f"Max items per collection (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--out-dir", "-o", default=".",
                        help="Output directory for PNG files (default: current dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph from ChromaDB…")
    G, node_collection, node_label = load_graph(
        threshold=args.threshold,
        sample=args.sample,
    )

    collections = sorted({node_collection[n] for n in G.nodes()})
    palette = collection_palette(collections)

    print(f"\nCollections : {collections}")
    print(f"Nodes       : {G.number_of_nodes()}")
    print(f"Edges       : {G.number_of_edges()}  (threshold={args.threshold})\n")

    draw_galaxy(G, node_collection, node_label, palette,
                out_path=str(out / "galaxy.png"))

    draw_heatmap(G, node_collection, node_label, palette,
                 out_path=str(out / "heatmap.png"))

    draw_chord(G, node_collection, palette,
               out_path=str(out / "chord.png"))

    print("\nDone!")
    print(f"  {out / 'galaxy.png'}   – radial spring layout, glowing nodes")
    print(f"  {out / 'heatmap.png'}  – cosine similarity matrix")
    print(f"  {out / 'chord.png'}    – cross-collection chord diagram")


if __name__ == "__main__":
    main()
