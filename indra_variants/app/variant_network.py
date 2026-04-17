import re
import math
import copy
import itertools as _it
import logging
import time as _time
import urllib.parse as _url
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache
from typing import Optional

from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, MATCH, ctx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indra_variants.app.config import DATA_DIR, PORT, DEBUG

cyto.load_extra_layouts()

TSV_RE = re.compile(r"^(?P<prot>.+)_variant_effects_with_clinvar_with_domains\.tsv$", re.I)
AA_SUB_RE = re.compile(r"^[A-Za-z\*]+(?P<pos>\d+)[A-Za-z\*=]+$")
P_DOT_RE = re.compile(r"p\.[A-Za-z\*]+(?P<pos>\d+)[A-Za-z\*=]+", re.I)


def _sort_text(value: str) -> str:
    return value.casefold()


def _alpha_bucket(value: str) -> str:
    if not value:
        return "#"
    first = value[0].upper()
    return first if "A" <= first <= "Z" else "#"


def _encode_route_value(value: str) -> str:
    return _url.quote(value, safe="")


def _decode_route_value(value: str) -> str:
    return _url.unquote(value)


def _protein_href(prot: str) -> str:
    return f"/protein/{_encode_route_value(prot)}"


def _endpoint_href(endpoint: str) -> str:
    return f"/endpoint/{_encode_route_value(endpoint)}"


TSV_FILES = {
    TSV_RE.match(p.name).group("prot"): p
    for p in Path(DATA_DIR).iterdir()
    if TSV_RE.match(p.name)
}
PROTS = sorted(TSV_FILES, key=_sort_text)
PROT_OPTIONS = [{'label': p, 'value': p} for p in PROTS]

def _build_endpoint_index() -> tuple[list[str], dict[str, dict[str, dict[str, int]]]]:
    endpoint_index: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    for prot, tsv_path in TSV_FILES.items():
        try:
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                usecols=["biological_process/disease", "variant_info"]
            ).fillna('')
        except ValueError:
            df = pd.read_csv(tsv_path, sep="\t").fillna('')
            if "biological_process/disease" not in df.columns:
                continue

        work = pd.DataFrame({
            "endpoint": df["biological_process/disease"].astype(str).str.strip(),
            "variant": (
                df["variant_info"].astype(str).str.strip()
                if "variant_info" in df.columns
                else pd.Series("", index=df.index)
            ),
        })
        work = work[work["endpoint"].ne("")]
        if work.empty:
            continue

        grouped = work.groupby("endpoint", sort=False).agg(
            row_count=("endpoint", "size"),
            variant_count=("variant", lambda s: s[s.ne("")].nunique()),
        )
        for endpoint_name, stats in grouped.iterrows():
            endpoint_index[endpoint_name][prot] = {
                "row_count": int(stats["row_count"]),
                "variant_count": int(stats["variant_count"]),
            }

    endpoint_names = sorted(endpoint_index, key=_sort_text)
    return endpoint_names, dict(endpoint_index)


ENDPOINTS, ENDPOINT_INDEX = _build_endpoint_index()
ENDPOINT_OPTIONS = [{'label': e, 'value': e} for e in ENDPOINTS]

# Reference coverage tables (publication / supplementary-style summary).
_STATS_BP_ROWS = [
    ("Cell population proliferation", 2206, 749, 1370, 1742),
    ("Apoptotic process", 1878, 667, 1252, 1495),
    ("DNA-templated transcription", 1113, 500, 883, 863),
    ("Cell death", 1105, 413, 763, 849),
    ("Neoplasm invasiveness", 812, 339, 567, 623),
    ("Localization", 776, 418, 679, 572),
    ("Cell differentiation", 749, 350, 538, 630),
    ("Cell growth", 699, 332, 526, 532),
    ("Cell survival", 695, 309, 528, 529),
    ("Cell migration", 570, 277, 472, 408),
]
_STATS_DISEASE_ROWS = [
    ("Neoplasms", 503, 231, 359, 382),
    ("Parkinson disease", 210, 37, 79, 156),
    ("Alzheimer disease", 143, 55, 100, 116),
    ("Melanoma", 75, 24, 37, 61),
    ("Breast neoplasms", 55, 33, 48, 42),
    ("Syndrome", 46, 29, 46, 28),
    ("Infections", 38, 31, 38, 35),
    ("Amyotrophic lateral sclerosis", 34, 4, 32, 6),
    ("Disease", 33, 28, 33, 26),
    ("Frontotemporal lobar degeneration", 32, 4, 32, 5),
]
# (gene, paths, variants, bps_or_diseases, pmids)
_STATS_GENE_ROWS = [
    ("TP53", 1298, 87, 110, 261),
    ("BRAF", 1218, 30, 130, 562),
    ("KRAS", 1152, 27, 162, 365),
    ("TARDBP", 909, 35, 54, 30),
    ("LRRK2", 398, 17, 77, 166),
    ("SOD1", 305, 16, 53, 141),
    ("JAK2", 231, 16, 50, 125),
    ("HRAS", 225, 14, 56, 89),
    ("MAPT", 187, 19, 77, 55),
    ("RPS6KB1", 183, 4, 58, 100),
    ("PIK3CA", 180, 9, 47, 55),
    ("EGFR", 179, 29, 38, 78),
    ("SNCA", 178, 10, 45, 52),
    ("NRAS", 173, 14, 47, 63),
    ("CTNNB1", 156, 20, 36, 59),
    ("DNM1L", 156, 16, 41, 48),
    ("RAC1", 151, 16, 51, 54),
    ("MDM2", 136, 22, 41, 37),
    ("GSK3B", 128, 15, 39, 40),
    ("PTEN", 127, 23, 39, 40),
]


def _stats_value_bp_disease(row: tuple, metric: str) -> int:
    _name, paths, genes, variants, pmids = row
    return {"path": paths, "gene": genes, "variant": variants,
            "pmid": pmids}[metric]


def _stats_value_gene(row: tuple, metric: str) -> int:
    _gene, paths, variants, bp_dis, pmids = row
    return {"path": paths, "gene": bp_dis, "variant": variants,
            "pmid": pmids}[metric]


def _stats_axis_label(metric: str, for_gene_chart: bool) -> str:
    if metric == "path":
        return "Paths"
    if metric == "variant":
        return "Variants"
    if metric == "pmid":
        return "PMIDs"
    return "BPs / diseases" if for_gene_chart else "Genes"


def _stats_network_href(label: str, for_gene_chart: bool) -> str:
    """Resolve bar label to an in-app network URL, or '' if not indexed."""
    key = (label or "").strip()
    if not key:
        return ""
    if for_gene_chart:
        for p in PROTS:
            if p.casefold() == key.casefold():
                return _protein_href(p)
        return ""
    for ep in ENDPOINT_INDEX:
        if ep.casefold() == key.casefold():
            return _endpoint_href(ep)
    return ""


def _stats_point_href(point: dict) -> Optional[str]:
    cd = point.get("customdata")
    if isinstance(cd, (list, tuple)) and len(cd) >= 1:
        href = cd[0]
        if isinstance(href, str) and href.startswith("/"):
            return href
    if isinstance(cd, str) and cd.startswith("/"):
        return cd
    return None


# Per-bar pixel height – kept constant so all three charts have the same bar size.
STATS_BAR_PX = 34
STATS_FIG_MARGIN_PX = 118   # title + x-axis + padding
# Convenience heights (used by both the figure and the card container)
STATS_FIG_HEIGHT_10 = STATS_BAR_PX * 10 + STATS_FIG_MARGIN_PX   # BP & disease
STATS_FIG_HEIGHT_20 = STATS_BAR_PX * 20 + STATS_FIG_MARGIN_PX   # gene (top 20)
STATS_FIG_HEIGHT = STATS_FIG_HEIGHT_10   # backward-compat alias


def _stats_bar_figure(
    rows: list[tuple],
    metric: str,
    title: str,
    for_gene_chart: bool,
    value_fn,
    *,
    bar_color: str,
    plot_bg: str = "#f8fafc",
    paper_bg: str = "rgba(0,0,0,0)",
    height: int = STATS_FIG_HEIGHT_10,
) -> go.Figure:
    scored = [(value_fn(r, metric), r) for r in rows]
    scored.sort(key=lambda t: t[0], reverse=True)
    values = [v for v, _ in scored]
    labels = [r[0] for _, r in scored]
    axis_title = _stats_axis_label(metric, for_gene_chart)
    hrefs = [_stats_network_href(lab, for_gene_chart) for lab in labels]
    hover_hint = [
        "Click to open graph" if h else "No matching graph in this build"
        for h in hrefs
    ]
    customdata = list(zip(hrefs, hover_hint))
    bar_height = 22
    fig = go.Figure(
        data=[go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=bar_color,
                line=dict(color="rgba(255,255,255,0.45)", width=1),
            ),
            text=values,
            textposition="outside",
            textfont=dict(color=U["ink_soft"], size=11),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x} " + axis_title + "<br>"
                "<span style='font-size:11px;color:#6f6b63'>%{customdata[1]}</span>"
                "<extra></extra>"
            ),
        )]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=U["ink"],
                                          family=U["font_display"])),
        xaxis_title=axis_title,
        yaxis=dict(autorange="reversed", title=""),
        margin=dict(l=10, r=88, t=52, b=44),
        height=height,
        font=dict(family=U["font_ui"], size=12, color=U["ink_soft"]),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        hoverlabel=dict(bgcolor=U["panel"], font_size=12,
                        font_family=U["font_ui"]),
    )
    fig.update_xaxes(
        gridcolor="rgba(45, 42, 36, 0.08)",
        zeroline=False,
        title_font=dict(color=U["muted"], size=12, family=U["font_ui"]),
    )
    fig.update_yaxes(tickfont=dict(size=11, color=U["ink_soft"],
                                   family=U["font_ui"]))
    return fig


def _build_alpha_directory(items, query: str, href_builder, columns: int = 3):
    query_norm = (query or "").strip().casefold()
    blocks = []
    bucket_order = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["#"]

    for letter in bucket_order:
        group = [item for item in items if _alpha_bucket(item) == letter]
        if query_norm:
            group = [item for item in group if query_norm in item.casefold()]
        if not group:
            continue

        summary_label = "Other" if letter == "#" else letter
        blocks.append(
            html.Details([
                html.Summary(f"{summary_label} ({len(group)})",
                             style={'cursor': 'pointer', 'fontSize': 20}),
                html.Ul([
                    html.Li(html.A(
                        item,
                        href=href_builder(item),
                        style={'textDecoration': 'none',
                               'color': U['link'],
                               'fontWeight': 'bold'
                               if query_norm and query_norm in item.casefold()
                               else 'normal'}))
                    for item in sorted(group, key=_sort_text)
                ], style={'columnCount': columns, 'listStyle': 'none',
                          'padding': 0, 'margin': '6px 0'})
            ], open=bool(query_norm))
        )

    if blocks:
        return blocks
    return html.Div("No results.",
                    style={'color': U['muted'], 'fontSize': 15,
                           'padding': '12px 0', 'fontFamily': U['font_ui']})



_log = logging.getLogger(__name__)

# ---------- LNS crossing minimization (Wilson et al., IEEE TVCG 2025) ------
import random as _rand


def _n_comb3(n):
    if n < 3:
        return 0
    return n * (n - 1) * (n - 2) // 6


def _add_anchors(layer_nodes, cross_layer_edges, node_layer):
    """Insert dummy anchor nodes for edges spanning >1 layer.

    Returns (aug_layers, adj_edges, node_layer) – the augmented graph
    where every edge connects adjacent layers.
    """
    aug = {li: list(ns) for li, ns in layer_nodes.items()}
    adj_edges: list[tuple] = []
    _aid = 0
    for src, tgt, w in cross_layer_edges:
        ls, lt = node_layer[src], node_layer[tgt]
        if ls > lt:
            src, tgt = tgt, src
            ls, lt = lt, ls
        if lt - ls == 1:
            adj_edges.append((src, tgt, w))
        else:
            prev = src
            for mid in range(ls + 1, lt):
                aname = f"__anch_{_aid}"
                _aid += 1
                aug.setdefault(mid, []).append(aname)
                node_layer[aname] = mid
                adj_edges.append((prev, aname, w))
                prev = aname
            adj_edges.append((prev, tgt, w))
    return aug, adj_edges


def _count_adj_crossings(order, adj_edges, node_layer, pos):
    """Count crossings between adjacent-layer edges using position dict."""
    ebl: dict[int, list] = defaultdict(list)
    for s, t, _w in adj_edges:
        ls = node_layer[s]
        ebl[ls].append((pos.get(s, 0.0), pos.get(t, 0.0)))
    total = 0
    for _li, edges in ebl.items():
        for a in range(len(edges)):
            for b in range(a + 1, len(edges)):
                if (edges[a][0] - edges[b][0]) * \
                   (edges[a][1] - edges[b][1]) < 0:
                    total += 1
    return total


def _optimize_layer_ordering(layer_nodes, cross_layer_edges,
                             time_budget=3.0, sub_time=0.3,
                             neighbourhood_k=12, fixed_layers=None):
    """Minimise edge crossings with Large Neighbourhood Search (LNS).

    1. Add anchor (dummy) nodes so every edge spans exactly one layer.
    2. Build a barycentric initial solution  (fast, O(n·iter)).
    3. Repeatedly pick a random candidate node, collect a small
       neighbourhood (≤ *neighbourhood_k* nodes per layer), solve
       that sub-problem optimally via ILP, and splice the improved
       ordering back into the global solution.
    4. Stop when *time_budget* seconds have elapsed.
    5. Strip anchor nodes from the final output.
    """
    fixed_layers = set(fixed_layers or [])
    node_layer: dict = {}
    for li, nodes in layer_nodes.items():
        for n in nodes:
            node_layer[n] = li

    aug, adj_edges = _add_anchors(layer_nodes, cross_layer_edges, node_layer)

    # Adjacency (including anchors) for barycentric
    adj: dict = defaultdict(set)
    for s, t, _w in adj_edges:
        adj[s].add(t)
        adj[t].add(s)

    # Edge weight lookup keyed on (lower-layer-node, upper-layer-node)
    edge_w: dict[tuple, float] = {}
    for s, t, w in adj_edges:
        edge_w[(s, t)] = edge_w.get((s, t), 0) + w

    sorted_layers = sorted(aug.keys())
    movable_nodes = [
        n for li in sorted_layers if li not in fixed_layers for n in aug[li]
    ]

    # ---------- Phase 1: barycentric initial solution ---------------------
    order: dict[int, list] = {li: list(ns) for li, ns in aug.items()}
    pos: dict = {}

    def _assign_pos(li):
        for i, n in enumerate(order[li]):
            pos[n] = float(i)

    for li in sorted_layers:
        _assign_pos(li)

    def _bary(n):
        vals = [pos[nb] for nb in adj[n] if nb in pos]
        return sum(vals) / len(vals) if vals else pos.get(n, 0.0)

    for _ in range(20):
        for li in sorted_layers:
            if li in fixed_layers:
                continue
            order[li].sort(key=lambda n: (_bary(n), n))
            _assign_pos(li)
        for li in reversed(sorted_layers):
            if li in fixed_layers:
                continue
            order[li].sort(key=lambda n: (_bary(n), n))
            _assign_pos(li)

    # ---------- Phase 2: LNS – local ILP improvements --------------------
    best_crossings = _count_adj_crossings(order, adj_edges, node_layer, pos)
    t0 = _time.time()
    n_improvements = 0
    n_iters = 0

    while movable_nodes and _time.time() - t0 < time_budget and best_crossings > 0:
        n_iters += 1
        candidate = _rand.choice(movable_nodes)
        c_layer = node_layer[candidate]

        # Neighbourhood: candidate + 1-hop + 2-hop, capped per layer
        sub_set: dict[int, set] = defaultdict(set)
        sub_set[c_layer].add(candidate)
        for nb in adj[candidate]:
            sub_set[node_layer[nb]].add(nb)
        for nb in list(adj[candidate]):
            for nb2 in adj[nb]:
                sub_set[node_layer[nb2]].add(nb2)

        sub_nodes: dict[int, list] = {}
        for li, ns in sub_set.items():
            if li in fixed_layers:
                continue
            lst = [n for n in order[li] if n in ns][:neighbourhood_k]
            if lst:
                sub_nodes[li] = lst

        if all(len(ns) < 2 for ns in sub_nodes.values()):
            continue

        # Expand each touched layer to include ALL its nodes (up to the cap).
        # Without this, two source nodes in the same layer (e.g. PTGS2 and
        # glucocorticoid) whose anchor chains only share a fixed ancestor
        # (layer 0 variants) are never simultaneously visible in a single
        # sub-problem, so the ILP cannot detect or fix their crossing edges.
        for li in list(sub_nodes.keys()):
            sub_nodes[li] = order[li][:neighbourhood_k]

        # Collect sub-edges (only between adjacent-layer sub-node pairs)
        sub_node_all = {n for ns in sub_nodes.values() for n in ns}
        sub_edges = [(s, t, w) for (s, t), w in edge_w.items()
                     if s in sub_node_all and t in sub_node_all]
        if not sub_edges:
            continue

        sub_result = _solve_sub_ilp(sub_nodes, sub_edges, node_layer,
                                    sub_time)
        if sub_result is None:
            continue

        # Splice improved sub-ordering back into global order
        old_order = {li: list(order[li]) for li in sub_result}
        for li, sub_ordered in sub_result.items():
            sset = set(sub_ordered)
            idx_map = sorted(i for i, n in enumerate(order[li]) if n in sset)
            base = [n for n in order[li] if n not in sset]
            for slot, n in zip(idx_map, sub_ordered):
                base.insert(slot, n)
            order[li] = base
            _assign_pos(li)

        new_crossings = _count_adj_crossings(order, adj_edges, node_layer, pos)
        if new_crossings < best_crossings:
            best_crossings = new_crossings
            n_improvements += 1
        else:
            for li, old in old_order.items():
                order[li] = old
                _assign_pos(li)

    elapsed = _time.time() - t0
    _log.info("LNS crossing minimisation: %d crossings, %d improvements, "
              "%d iters in %.2fs", best_crossings, n_improvements,
              n_iters, elapsed)

    # Strip anchors, return only original-node orderings
    result: dict[int, list] = {}
    for li in sorted(layer_nodes.keys()):
        result[li] = [n for n in order[li]
                      if not str(n).startswith("__anch_")]
    return result


def _solve_sub_ilp(layer_nodes, edges, node_layer, cutoff):
    """Solve a small sub-problem exactly via ILP.  Returns optimised
    orderings or *None* on failure."""
    all_n = []
    for li in sorted(layer_nodes):
        all_n.extend(layer_nodes[li])
    if len(all_n) < 2:
        return None
    iid = {n: i for i, n in enumerate(all_n)}

    adj_e: list[tuple] = []
    for s, t, w in edges:
        ls, lt = node_layer[s], node_layer[t]
        if ls == lt:
            continue
        if ls > lt:
            s, t = t, s
        adj_e.append((iid[s], iid[t], w))

    x_vars: dict[tuple, int] = {}
    nv = 0
    for li in sorted(layer_nodes):
        ids = sorted(iid[n] for n in layer_nodes[li])
        for a, b in _it.combinations(ids, 2):
            x_vars[(a, b)] = nv
            nv += 1

    # Group edges by (src_layer, tgt_layer) so only same-span pairs interact
    ebl: dict[tuple, list] = defaultdict(list)
    for s, t, w in adj_e:
        ls, lt = node_layer[all_n[s]], node_layer[all_n[t]]
        ebl[(min(ls, lt), max(ls, lt))].append((s, t, w))

    c_vars: dict = {}
    c_wt: dict = {}
    for _lpair, elist in ebl.items():
        for i in range(len(elist)):
            u1, v1, w1 = elist[i]
            for j in range(i + 1, len(elist)):
                u2, v2, w2 = elist[j]
                if u1 != u2 and v1 != v2:
                    c_vars[((u1, v1), (u2, v2))] = nv
                    c_wt[nv] = w1 * w2
                    nv += 1

    if not c_vars:
        return None

    n_trans = sum(
        2 * _n_comb3(len(layer_nodes[li]))
        for li in sorted(layer_nodes)
    )
    n_cons = n_trans + 2 * len(c_vars)
    A = lil_matrix((n_cons, nv))
    b_ub = [0.0] * n_cons
    r = 0

    for li in sorted(layer_nodes):
        ids = sorted(iid[n] for n in layer_nodes[li])
        for i, j, k in _it.combinations(ids, 3):
            A[r, x_vars[(i, j)]] = -1
            A[r, x_vars[(j, k)]] = -1
            A[r, x_vars[(i, k)]] = 1
            r += 1
            A[r, x_vars[(i, j)]] = 1
            A[r, x_vars[(j, k)]] = 1
            A[r, x_vars[(i, k)]] = -1
            b_ub[r] = 1
            r += 1

    def _xdir(a, b_id):
        if (a, b_id) in x_vars:
            return x_vars[(a, b_id)], 1, 0
        return x_vars[(b_id, a)], -1, 1

    for ((u1, v1), (u2, v2)), ci in c_vars.items():
        xi_u, du, fu = _xdir(u1, u2)
        xi_v, dv, fv = _xdir(v1, v2)
        A[r, xi_u] = du;  A[r, xi_v] = -dv;  A[r, ci] = -1
        b_ub[r] = fv - fu
        r += 1
        A[r, xi_u] = -du; A[r, xi_v] = dv;  A[r, ci] = -1
        b_ub[r] = fu - fv
        r += 1

    obj = [0.0] * nv
    for vi, w in c_wt.items():
        obj[vi] = w

    try:
        res = linprog(
            obj, method="highs",
            A_ub=A.tocsc(), b_ub=b_ub,
            bounds=[(0, 1)] * nv,
            integrality=[1] * nv,
            options={"time_limit": cutoff, "disp": False},
        )
    except Exception:
        return None
    if res.x is None:
        return None

    result: dict[int, list] = {}
    for li in sorted(layer_nodes):
        ids = sorted(iid[n] for n in layer_nodes[li])
        if len(ids) < 2:
            result[li] = [all_n[ids[0]]] if ids else []
            continue
        rank = {n: 0 for n in ids}
        for a, b_id in _it.combinations(ids, 2):
            if round(res.x[x_vars[(a, b_id)]]) == 1:
                rank[b_id] += 1
            else:
                rank[a] += 1
        ordered = sorted(ids, key=lambda n: rank[n])
        result[li] = [all_n[n] for n in ordered]
    return result


def format_star_rating(star_val):
    review_map = {
        4.0: "practice guideline",
        3.0: "expert panel",
        2.0: "multiple submitters, no conflicts",
        1.0: "single submitter",
        0.0: "no assertion criteria provided" # Updated for clarity
    }
    try:
        s_val = float(star_val)
        num_stars = int(s_val)
        review_text = review_map.get(s_val, "review status not specified")
        
        if num_stars > 0:
            return f"{'★ ' * num_stars} ({review_text})"
        else:
            return f"({review_text})"
            
    except (ValueError, TypeError):
        return "(no review info)"


def _variant_aa_position(variant_label: str, name_label: str = "") -> Optional[int]:
    m = AA_SUB_RE.match((variant_label or "").strip())
    if m:
        return int(m.group("pos"))
    m = P_DOT_RE.search((name_label or "").strip())
    if m:
        return int(m.group("pos"))
    return None


# ----------------------Build Graph--------------------------–
def build_elements(prot: str, variant_aa_range: Optional[tuple[float, float]] = None):
    df_path = TSV_FILES[prot]
    df = pd.read_csv(df_path, sep="\t").fillna('')

    def choose_best_clinvar(existing: Optional[dict], candidate: Optional[dict]):
        if not candidate:
            return existing
        if not existing:
            return candidate
        # Prefer richer condition lists, then keep existing.
        prev_len = len(existing.get("conditions", ""))
        cand_len = len(candidate.get("conditions", ""))
        if cand_len > prev_len:
            return candidate
        return existing

    G = nx.MultiDiGraph()
    G.add_node(prot)
    variants = set(df["variant_info"])
    endpoints = set(df["biological_process/disease"])
    endpoint_freq = df["biological_process/disease"].value_counts().to_dict()
    protein_uniprot_id = ""

    variant_meta = defaultdict(lambda: {
        "domains": set(),
        "domain_notes": set(),
        "clinvar_data": None,
        "protein_pos": None,
        "allele_id": None,
        "dbsnp_rs": None,
    })
    edge_bucket = defaultdict(lambda: {
        "pmids": set(),
        "notes": set(),
        "clinvar_data": None,
        "count": 0
    })
    chain_pos: dict[str, int] = {}

    for _, row in df.iterrows():
        var = row["variant_info"]
        name_label = row.get("Name", "")
        if not protein_uniprot_id:
            domain_protein_id = str(row.get("DomainProteinID", "")).strip()
            if domain_protein_id and domain_protein_id.lower() != "nan":
                protein_uniprot_id = domain_protein_id.split(";")[0].strip()
        protein_pos = _variant_aa_position(var, name_label)
        if protein_pos is not None:
            prev = variant_meta[var]["protein_pos"]
            variant_meta[var]["protein_pos"] = protein_pos if prev is None else min(prev, protein_pos)

        all_conditions = []
        for i in range(1, 11):
            disease = row.get(f"disease_{i}", "")
            if disease and 'not provided' not in disease.lower():
                all_conditions.append(disease)
        clinvar_data = None
        if all_conditions:
            clinvar_data = {
                "pathogenicity": row.get("significance_1", "N/A"),
                "review": format_star_rating(row.get("star_1", 0.0)),
                "conditions": "; ".join(all_conditions)
            }
        variant_meta[var]["clinvar_data"] = choose_best_clinvar(variant_meta[var]["clinvar_data"], clinvar_data)
        allele_id = str(row.get("#AlleleID", "")).strip()
        if allele_id and allele_id.lower() != "nan":
            try:
                variant_meta[var]["allele_id"] = str(int(float(allele_id)))
            except (TypeError, ValueError):
                variant_meta[var]["allele_id"] = allele_id
        dbsnp_rs = str(row.get("RS# (dbSNP)", "")).strip()
        if dbsnp_rs and dbsnp_rs.lower() != "nan":
            try:
                variant_meta[var]["dbsnp_rs"] = str(int(float(dbsnp_rs)))
            except (TypeError, ValueError):
                variant_meta[var]["dbsnp_rs"] = dbsnp_rs

        # Keep domain information on variants, but hide domain nodes in layout.
        features = [f.strip() for f in (row.get("DomainFeature", "") or "").split(';') if f.strip()]
        notes = [n.strip() for n in (row.get("DomainNote", "") or "").split(';') if n.strip()]
        for d in features:
            if d != "CHAIN":
                variant_meta[var]["domains"].add(d)
        for note in notes:
            if note and note != "1":
                variant_meta[var]["domain_notes"].add(note)

        # Baseline edge: protein -> variant
        pv_key = (prot, var, "PV")
        edge_bucket[pv_key]["count"] += 1
        pmid_val = str(row.get("pmid", "")).strip()
        if pmid_val:
            edge_bucket[pv_key]["pmids"].add(pmid_val)
        edge_bucket[pv_key]["clinvar_data"] = choose_best_clinvar(
            edge_bucket[pv_key]["clinvar_data"], variant_meta[var]["clinvar_data"]
        )

        # Causal chain from variant onwards.
        src = var
        _hop = 0
        for seg in str(row.get("chain", "")).split(" -[")[1:]:
            if "]->" not in seg:
                continue
            rel, tgt = seg.split("]->", 1)
            rel = rel.strip()
            tgt = tgt.strip()
            if not tgt:
                continue
            _hop += 1
            chain_pos[tgt] = max(chain_pos.get(tgt, 0), _hop)
            key = (src, tgt, rel)
            edge_bucket[key]["count"] += 1
            pmid = str(row.get("pmid", "")).strip()
            if pmid:
                edge_bucket[key]["pmids"].add(pmid)
            edge_bucket[key]["clinvar_data"] = choose_best_clinvar(edge_bucket[key]["clinvar_data"], clinvar_data)
            src = tgt

    for (u, v, rel), payload in edge_bucket.items():
        edge_attrs = {"relation": rel, "weight": payload["count"]}
        if payload["pmids"]:
            edge_attrs["pmid"] = "; ".join(sorted(payload["pmids"]))
        if payload["notes"]:
            edge_attrs["note"] = "; ".join(sorted(payload["notes"]))
        if payload["clinvar_data"]:
            edge_attrs["clinvar_data"] = payload["clinvar_data"]
        G.add_edge(u, v, **edge_attrs)

    subgraph_data: dict = {}


    if variant_aa_range is not None:
        lo, hi = float(variant_aa_range[0]), float(variant_aa_range[1])
        exclude = set()
        for v in variants:
            if not G.has_node(v):
                continue
            vm = variant_meta.get(v, {})
            p = vm.get("protein_pos")
            if p is not None:
                try:
                    pi = int(p)
                except (TypeError, ValueError):
                    pi = None
            else:
                pi = None
            if pi is None:
                po = _variant_aa_position(v, "")
                pi = int(po) if po is not None else None
            if pi is None or pi < lo or pi > hi:
                exclude.add(v)
        for v in exclude:
            if G.has_node(v):
                G.remove_node(v)
        if G.has_node(prot):
            seen = {prot}
            dq = deque([prot])
            while dq:
                u = dq.popleft()
                for _, v, _ in G.out_edges(u, data=True):
                    if v not in seen:
                        seen.add(v)
                        dq.append(v)
            for n in list(G.nodes()):
                if n not in seen:
                    G.remove_node(n)
        variants = {v for v in variants if G.has_node(v)}
        endpoints = {e for e in endpoints if G.has_node(e)}
        for k in list(chain_pos.keys()):
            if not G.has_node(k):
                del chain_pos[k]

    # ---------- Kind-aware layered layout with pseudo nodes ----------
    # Layer 0: protein + variants (fixed)
    # Layer -1: endpoints reachable ONLY directly from variants
    # Layers 1..N: longest-path depth plus extra spacing for repeated kinds
    # (e.g. protein -> protein or endpoint -> endpoint).

    def _node_kind(n: str) -> str:
        if n in variants:
            return "variant"
        if n in endpoints:
            return "endpoint"
        return "intermediate"

    node_kind = {n: _node_kind(n) for n in G.nodes()}

    # -- Step 1: identify "direct-only" endpoints → layer -1 ------------
    # An endpoint qualifies as "direct-only" when ALL its graph neighbours
    # (both predecessors and successors, excluding the root protein) are
    # variants.  If it also feeds into intermediates or other endpoints it
    # participates in longer chains and must go through normal layering.
    _direct_only_nodes: dict[str, int] = {}

    def _is_direct_only(node: str) -> bool:
        preds = {u for u, _, _ in G.in_edges(node, data=True) if u != prot}
        succs = {v for _, v, _ in G.out_edges(node, data=True)}
        if not preds:
            return False
        all_variant_preds = all(p in variants for p in preds)
        has_non_variant_succs = any(
            s not in variants and s != prot for s in succs
        )
        return all_variant_preds and not has_non_variant_succs

    for ep in endpoints:
        if _is_direct_only(ep):
            _direct_only_nodes[ep] = -1
    _direct_only_eps = {n for n in _direct_only_nodes if n in endpoints}

    # -- Step 2: layer seeds from chain positions -------------------------
    node_depth: dict[str, int] = {}
    for v in variants:
        node_depth[v] = 0
    node_depth[prot] = 0

    _init_depth: dict[str, int] = {}
    for n in G.nodes():
        if n == prot or n in variants or n in _direct_only_nodes:
            continue
        if n in chain_pos:
            _init_depth[n] = chain_pos[n]
        else:
            _init_depth[n] = 1

    # -- Step 3: condensation DAG propagation (cycle-safe) ---------------
    #    Collapse SCCs so cycles don't cascade layer depths, then align
    #    later kinds after the deepest preceding layer of the source kind.
    _non_vp = [
        n for n in G.nodes()
        if n != prot and n not in variants and n not in _direct_only_nodes
    ]
    if _non_vp:
        _H = nx.DiGraph()
        for n in _non_vp:
            _H.add_node(n)
        for u, v, _ in G.edges(data=True):
            if u in _H and v in _H:
                _H.add_edge(u, v)
        _C = nx.condensation(_H)
        _scc_map = _C.graph["mapping"]
        _scc_depth: dict[int, int] = {}
        for n in _non_vp:
            sid = _scc_map[n]
            _scc_depth[sid] = max(_scc_depth.get(sid, 0),
                                  _init_depth.get(n, 1))
        _topo_sids = list(nx.topological_sort(_C))
        for sid in _topo_sids:
            for succ_sid in _C.successors(sid):
                if _scc_depth.get(succ_sid, 0) <= _scc_depth.get(sid, 0):
                    _scc_depth[succ_sid] = _scc_depth[sid] + 1

        for n in _non_vp:
            node_depth[n] = _scc_depth[_scc_map[n]]

        # Rewrite endpoint depths: use only within-endpoint distance
        # so that all endpoints directly reachable from intermediates
        # land on the same (first) endpoint layer, and only
        # endpoint-to-endpoint chains create additional endpoint layers.
        _ep_nodes = {n for n in _non_vp if node_kind[n] == "endpoint"}
        if _ep_nodes:
            _ep_depth: dict[str, int] = {}
            for n in _ep_nodes:
                has_non_ep_pred = any(
                    node_kind.get(u, "variant") != "endpoint"
                    for u, _, _ in G.in_edges(n, data=True)
                )
                _ep_depth[n] = 0 if has_non_ep_pred else 1

            _ep_sub = nx.DiGraph()
            for n in _ep_nodes:
                _ep_sub.add_node(n)
            for u, v, _ in G.edges(data=True):
                if u in _ep_nodes and v in _ep_nodes:
                    _ep_sub.add_edge(u, v)

            if _ep_sub.edges:
                _ep_C = nx.condensation(_ep_sub)
                _ep_scc = _ep_C.graph["mapping"]
                _ep_scc_d: dict[int, int] = {}
                for n in _ep_nodes:
                    sid = _ep_scc[n]
                    _ep_scc_d[sid] = min(
                        _ep_scc_d.get(sid, 999), _ep_depth[n])
                for sid in nx.topological_sort(_ep_C):
                    for succ in _ep_C.successors(sid):
                        if _ep_scc_d.get(succ, 0) <= _ep_scc_d[sid]:
                            _ep_scc_d[succ] = _ep_scc_d[sid] + 1
                for n in _ep_nodes:
                    _ep_depth[n] = _ep_scc_d[_ep_scc[n]]

            for n in _ep_nodes:
                node_depth[n] = _ep_depth[n]

        # Kind-zone separation: shift each kind so zones don't overlap.
        # Canonical order: intermediate first, then endpoint.
        # When the kind-feed graph has cycles (e.g. endpoint <-> intermediate),
        # fall back to the canonical order instead of skipping the shift.
        _kind_layers: dict[str, set] = defaultdict(set)
        for n in _non_vp:
            _kind_layers[node_kind[n]].add(node_depth[n])

        _kind_feeds: set = set()
        for u, v, _ in G.edges(data=True):
            ku, kv = node_kind.get(u, ""), node_kind.get(v, "")
            if ku and kv and ku != kv and ku != "variant" and kv != "variant":
                _kind_feeds.add((ku, kv))

        _CANONICAL_KIND_ORDER = ["intermediate", "endpoint"]

        if _kind_feeds:
            _kind_dag = nx.DiGraph(list(_kind_feeds))
            if _kind_dag.nodes and nx.is_directed_acyclic_graph(_kind_dag):
                _ordered_kinds = list(nx.topological_sort(_kind_dag))
            else:
                _ordered_kinds = [k for k in _CANONICAL_KIND_ORDER
                                  if k in _kind_layers]

            _kind_shift: dict[str, int] = defaultdict(int)
            for i in range(len(_ordered_kinds) - 1):
                src_k = _ordered_kinds[i]
                dst_k = _ordered_kinds[i + 1]
                if not _kind_layers.get(src_k) or not _kind_layers.get(dst_k):
                    continue
                src_max = max(_kind_layers[src_k]) + _kind_shift[src_k]
                dst_min = min(_kind_layers[dst_k]) + _kind_shift[dst_k]
                if dst_min <= src_max:
                    _kind_shift[dst_k] += src_max + 1 - dst_min

            for n in _non_vp:
                node_depth[n] += _kind_shift.get(node_kind[n], 0)
    else:
        for n, d in _init_depth.items():
            node_depth[n] = d

    node_depth.update(_direct_only_nodes)

    # Safety net: no endpoint (except direct-only at -1) may share layer 0
    # with variants, and no endpoint may share a layer with intermediates.
    _max_int_depth = max(
        (node_depth[n] for n in node_depth
         if node_kind.get(n) == "intermediate"),
        default=0,
    )
    _min_ep_floor = max(_max_int_depth + 1, 1)
    for n in list(node_depth):
        if node_kind.get(n) == "endpoint" and n not in _direct_only_nodes:
            if node_depth[n] < _min_ep_floor:
                node_depth[n] = _min_ep_floor

    def get_layer(n):
        if n == prot or n in variants:
            return 0
        return node_depth.get(n, 1)

    # -- Step 4: collect all layers -------------------------------------
    layers: dict[int, list] = defaultdict(list)
    for n in G.nodes():
        layers[get_layer(n)].append(n)

    def _variant_sort_key(variant_name: str):
        protein_pos = variant_meta[variant_name]["protein_pos"]
        return (
            protein_pos is None,
            protein_pos if protein_pos is not None else math.inf,
            variant_name,
        )

    if 0 in layers:
        ordered_variants = sorted(
            [n for n in layers[0] if n in variants],
            key=_variant_sort_key,
        )
        layer_zero_other = [n for n in layers[0] if n not in variants and n != prot]
        layers[0] = ordered_variants + layer_zero_other + ([prot] if prot in layers[0] else [])

    # -- Step 5: crossing minimisation (LNS) ---------------------------
    _cross = {}
    for u, v, _d in G.edges(data=True):
        lu, lv = get_layer(u), get_layer(v)
        if lu == lv:
            continue
        pair = (u, v) if lu < lv else (v, u)
        _cross[pair] = _cross.get(pair, 0) + 1
    weighted_cross = [(u, v, w) for (u, v), w in _cross.items()]

    optimized = _optimize_layer_ordering(
        dict(layers),
        weighted_cross,
        fixed_layers={0},
    )

    # -- Step 6: assign positions from optimised orderings --------------
    x_pos: dict[str, float] = {}
    sorted_layer_keys = sorted(optimized.keys())
    max_layer = max(sorted_layer_keys) if sorted_layer_keys else 0
    min_layer = min(sorted_layer_keys) if sorted_layer_keys else 0
    layer_gap = 190.0
    y_pos = {
        li: li * layer_gap
        for li in range(min_layer, max_layer + 1)
    }

    for li, ordered in optimized.items():
        if li == 0 and prot in ordered:
            ordered = [n for n in ordered if n != prot]
            ordered.append(prot)
        n_nodes = len(ordered)
        sp = 160.0 if li == 0 else 210.0
        start = -((n_nodes - 1) * sp) / 2.0
        for i, n in enumerate(ordered):
            x_pos[n] = start + i * sp

    for n in G.nodes():
        if n not in x_pos:
            x_pos[n] = 0.0

    # -- Step 6.5: Barycentric repositioning for non-variant layers ----------
    # The uniform even-spacing above looks artificially symmetric.  Reposition
    # each node to the centroid of its "toward-layer-0" parents so the layout
    # reflects actual connection structure, while preserving the crossing-
    # minimised ordering the LNS found.  Process layers in order of increasing
    # distance from layer 0 so parent positions are finalised first.
    _max_half_px = max(
        (len(nodes) - 1) * 170.0 / 2.0
        for nodes in optimized.values() if len(nodes) > 1
    ) if any(len(v) > 1 for v in optimized.values()) else 170.0

    # For each node, collect the adjacent-layer neighbors that are closer to
    # layer 0 (the variant layer).  Those are the "parents" that anchor x.
    _parent_adj: dict[str, list] = defaultdict(list)
    for _u, _v, _d in G.edges(data=True):
        _lu, _lv = get_layer(_u), get_layer(_v)
        if _lu == _lv:
            continue
        if _lu == 0:
            _parent_adj[_v].append(_u)
        elif _lv == 0:
            _parent_adj[_u].append(_v)
        elif abs(_lu) < abs(_lv):
            _parent_adj[_v].append(_u)
        else:
            _parent_adj[_u].append(_v)

    _bary_min_sep = 80.0
    for _li in sorted((li for li in optimized if li != 0), key=abs):
        _ordered = list(optimized[_li])
        if not _ordered:
            continue
        # Target x = centroid of parents already placed in x_pos
        _bx: dict[str, float] = {}
        for _nd in _ordered:
            _nbrs = [_nb for _nb in _parent_adj.get(_nd, []) if _nb in x_pos]
            _bx[_nd] = (
                sum(x_pos[_nb] for _nb in _nbrs) / len(_nbrs)
                if _nbrs else x_pos.get(_nd, 0.0)
            )
        # Keep LNS order; just slide each node toward its centroid x
        _xs = [max(-_max_half_px, min(_max_half_px, _bx[_nd])) for _nd in _ordered]
        # Forward pass: enforce minimum separation
        for _idx in range(1, len(_xs)):
            if _xs[_idx] < _xs[_idx - 1] + _bary_min_sep:
                _xs[_idx] = _xs[_idx - 1] + _bary_min_sep
        # Shift left if right boundary exceeded
        if _xs and _xs[-1] > _max_half_px:
            _shift = _xs[-1] - _max_half_px
            _xs = [_x - _shift for _x in _xs]
        # Backward pass: fix left-side violations after the shift
        for _idx in range(len(_xs) - 2, -1, -1):
            if _xs[_idx] > _xs[_idx + 1] - _bary_min_sep:
                _xs[_idx] = _xs[_idx + 1] - _bary_min_sep
        for _nd, _x in zip(_ordered, _xs):
            x_pos[_nd] = _x

    pos = {n: (x_pos[n], y_pos.get(get_layer(n), 0.0)) for n in G.nodes()}

    def _css_safe(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9_-]', '-', name)

    def _short_label(text, max_len=28):
        if len(text) <= max_len:
            return text
        cut = text[:max_len].rfind(' ')
        if cut < max_len // 2:
            cut = max_len
        return text[:cut].rstrip() + "..."

    raw_rel_types = sorted({
        d['relation'] for _, _, d in G.edges(data=True)
        if d['relation'] not in {'PV', 'DV', 'has_domain'}
    })
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    rel_color_safe = {_css_safe(r): palette[i % len(palette)] for i, r in enumerate(raw_rel_types)}
    rel_display = {_css_safe(r): r for r in raw_rel_types}

    els = []
    for n, (x, y) in pos.items():
        layer = get_layer(n)

        if n == prot:
            size = 90
        elif n in endpoints or n in _direct_only_eps:
            size = 52 + min(24, 5 * endpoint_freq.get(n, 1))
        else:
            size = 54

        if n == prot:
            label = n
            role_class = "role-protein"
        elif n in variants:
            label = n
            role_class = "role-variant"
        elif n in endpoints or n in _direct_only_eps:
            label = _short_label(n)
            role_class = "role-endpoint"
        else:
            label = _short_label(n)
            role_class = "role-intermediate"

        node_el = {
            "data": {
                "id": n,
                "label": label,
                "real": n,
                "role": role_class.replace("role-", ""),
                "motif_type": "",
                "motif_members": "",
                "domain_notes": "; ".join(sorted(variant_meta[n]["domain_notes"])) if n in variant_meta else ""
            },
            "classes": f"L{layer} {role_class}",
            "style": {"width": size, "height": size}
        }
        if n == prot:
            node_el["data"]["uniprot_id"] = protein_uniprot_id or prot
        if n in variants and n in variant_meta:
            vm = variant_meta[n]
            node_el["data"]["gene_symbol"] = prot
            if vm.get("protein_pos") is not None:
                node_el["data"]["protein_pos"] = vm["protein_pos"]
            if vm["allele_id"]:
                node_el["data"]["clinvar_allele"] = vm["allele_id"]
            if vm["dbsnp_rs"]:
                node_el["data"]["dbsnp_rs"] = vm["dbsnp_rs"]
            if vm["clinvar_data"]:
                node_el["data"]["clinvar_data"] = vm["clinvar_data"]

        if n in pos:
            node_el["position"] = {"x": pos[n][0], "y": pos[n][1]}
        els.append(node_el)

    for u, v, d in G.edges(data=True):
        relation = d.get('relation', '')
        if relation == 'PV': cls = 'edge-PV'
        elif relation == 'DV': cls = 'edge-DV'
        elif relation == 'has_domain': cls = 'edge-has_domain'
        else: cls = f"edge-{_css_safe(relation)}"

        src4indra = prot if u in variants else u
        edge_data = {
            "id": f"{u}->{v}_{d.get('pmid', '')}_{d.get('note', '')}",
            "source": u,
            "target": v,
            "rel": relation,
            "src4indra": src4indra,
        }
        if 'pmid' in d and d['pmid']:
            edge_data['pmid'] = d['pmid']
        if 'note' in d and d['note']:
            edge_data['note'] = d['note']
        if 'clinvar_data' in d and d['clinvar_data']:
            edge_data['clinvar_data'] = d['clinvar_data']
        if 'weight' in d:
            edge_data['evidence_count'] = d['weight']

        els.append({"data": edge_data, "classes": cls})

    edge_set = {(u, v) for u, v, _ in G.edges(data=True)}

    legend_rels = ['Gene–variant'] + raw_rel_types
    legend_colors = {
        'Gene–variant': '#c9c4bf',
        **{rel_display.get(k, k): v for k, v in rel_color_safe.items()}
    }

    return els, legend_rels, legend_colors, rel_color_safe, list(edge_set), subgraph_data


def build_endpoint_elements(endpoint: str):
    return copy.deepcopy(_build_endpoint_elements_cached(endpoint))


@lru_cache(maxsize=32)
def _build_endpoint_elements_cached(endpoint: str):
    prot_stats = ENDPOINT_INDEX.get(endpoint, {})
    if not prot_stats:
        return [], [], {}, {}, [], {}

    def _css_safe(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9_-]', '-', name)

    def _short_label(text, max_len=28):
        if len(text) <= max_len:
            return text
        cut = text[:max_len].rfind(' ')
        if cut < max_len // 2:
            cut = max_len
        return text[:cut].rstrip() + "..."

    def choose_best_clinvar(existing: Optional[dict], candidate: Optional[dict]):
        if not candidate:
            return existing
        if not existing:
            return candidate
        prev_len = len(existing.get("conditions", ""))
        cand_len = len(candidate.get("conditions", ""))
        if cand_len > prev_len:
            return candidate
        return existing

    protein_nodes: dict[str, dict] = {}
    variant_nodes: dict[str, dict] = {}
    intermediate_nodes: dict[str, dict] = {}
    edge_bucket: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0, "pmids": set(), "notes": set(),
    })
    edge_set: set[tuple[str, str]] = set()

    read_columns = [
        "biological_process/disease",
        "variant_info",
        "chain",
        "pmid",
        "#AlleleID",
        "RS# (dbSNP)",
        "significance_1",
        "star_1",
        *[f"disease_{i}" for i in range(1, 11)],
    ]

    for prot in sorted(prot_stats, key=_sort_text):
        tsv_path = TSV_FILES[prot]
        try:
            df = pd.read_csv(tsv_path, sep="\t", usecols=read_columns).fillna('')
        except ValueError:
            df = pd.read_csv(tsv_path, sep="\t").fillna('')

        if "biological_process/disease" not in df.columns or "variant_info" not in df.columns:
            continue

        endpoint_series = df["biological_process/disease"].astype(str).str.strip()
        sub = df[endpoint_series.eq(endpoint)]
        if sub.empty:
            continue

        protein_entry = protein_nodes.setdefault(prot, {
            "row_count": 0,
            "variant_ids": set(),
        })

        for _, row in sub.iterrows():
            variant_label = str(row.get("variant_info", "")).strip()
            if not variant_label:
                continue

            variant_id = f"{prot}::{variant_label}"
            protein_entry["row_count"] += 1
            protein_entry["variant_ids"].add(variant_id)

            variant_entry = variant_nodes.setdefault(variant_id, {
                "label": variant_label,
                "real": f"{prot} {variant_label}",
                "protein": prot,
                "row_count": 0,
                "pmids": set(),
                "clinvar_data": None,
                "allele_id": None,
                "dbsnp_rs": None,
            })
            variant_entry["row_count"] += 1

            pmid_val = str(row.get("pmid", "")).strip()
            if pmid_val:
                variant_entry["pmids"].add(pmid_val)

            all_conditions = []
            for i in range(1, 11):
                disease = str(row.get(f"disease_{i}", "")).strip()
                if disease and 'not provided' not in disease.lower():
                    all_conditions.append(disease)
            clinvar_data = None
            if all_conditions:
                clinvar_data = {
                    "pathogenicity": row.get("significance_1", "N/A"),
                    "review": format_star_rating(row.get("star_1", 0.0)),
                    "conditions": "; ".join(all_conditions),
                }
            variant_entry["clinvar_data"] = choose_best_clinvar(
                variant_entry["clinvar_data"], clinvar_data
            )

            allele_id = str(row.get("#AlleleID", "")).strip()
            if allele_id and allele_id.lower() != "nan":
                try:
                    variant_entry["allele_id"] = str(int(float(allele_id)))
                except (TypeError, ValueError):
                    variant_entry["allele_id"] = allele_id

            dbsnp_rs = str(row.get("RS# (dbSNP)", "")).strip()
            if dbsnp_rs and dbsnp_rs.lower() != "nan":
                try:
                    variant_entry["dbsnp_rs"] = str(int(float(dbsnp_rs)))
                except (TypeError, ValueError):
                    variant_entry["dbsnp_rs"] = dbsnp_rs

            pv_key = (prot, variant_id, "PV")
            edge_bucket[pv_key]["count"] += 1
            if pmid_val:
                edge_bucket[pv_key]["pmids"].add(pmid_val)
            edge_set.add((prot, variant_id))

            chain_str = str(row.get("chain", ""))
            segs = chain_str.split(" -[")[1:]
            src = variant_id
            for seg in segs:
                if "]->" not in seg:
                    continue
                rel, tgt = seg.split("]->", 1)
                rel = rel.strip()
                tgt = tgt.strip()
                if not tgt:
                    continue

                if tgt == endpoint:
                    tgt_id = endpoint
                elif tgt == prot:
                    tgt_id = prot
                elif tgt in prot_stats:
                    # This target also has its own variants for this disease,
                    # so it already exists as a protein node.  Give it a
                    # distinct intermediate-role ID so both can coexist.
                    tgt_id = f"{tgt}::inode"
                    if tgt_id not in intermediate_nodes:
                        intermediate_nodes[tgt_id] = {
                            "label": _short_label(tgt),
                            "real": tgt,
                            "row_count": 0,
                            "is_known_protein": True,
                        }
                    intermediate_nodes[tgt_id]["row_count"] += 1
                else:
                    tgt_id = tgt
                    if tgt_id not in intermediate_nodes:
                        is_known_prot = tgt_id in prot_stats or tgt_id in TSV_FILES
                        intermediate_nodes[tgt_id] = {
                            "label": _short_label(tgt_id),
                            "real": tgt_id,
                            "row_count": 0,
                            "is_known_protein": is_known_prot,
                        }
                    intermediate_nodes[tgt_id]["row_count"] += 1

                ekey = (src, tgt_id, rel)
                edge_bucket[ekey]["count"] += 1
                if pmid_val:
                    edge_bucket[ekey]["pmids"].add(pmid_val)
                edge_set.add((src, tgt_id))
                src = tgt_id

    ordered_proteins = sorted(
        protein_nodes.items(),
        key=lambda kv: (-len(kv[1]["variant_ids"]), -kv[1]["row_count"], kv[0].casefold())
    )
    ordered_variants = sorted(
        variant_nodes.items(),
        key=lambda kv: (-kv[1]["row_count"], kv[1]["protein"].casefold(), kv[1]["label"].casefold())
    )

    total_proteins = len(ordered_proteins)
    total_variants = len(ordered_variants)
    total_records = sum(info["row_count"] for info in protein_nodes.values())

    raw_rel_types = sorted({
        rel for (_s, _t, rel) in edge_bucket if rel != "PV"
    })
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    rel_color_safe = {_css_safe(r): palette[i % len(palette)] for i, r in enumerate(raw_rel_types)}
    rel_display = {_css_safe(r): r for r in raw_rel_types}

    # --- Layered layout --------------------------------------------------
    # Indirect paths (gene→variant→intermediates→disease) go BELOW disease.
    # Direct paths (gene→variant→disease, no intermediates) go ABOVE disease.
    #
    # Bottom section:  layer 0  = indirect genes
    #                  layer 1  = indirect variants
    #                  layer 2+ = intermediate nodes (longest-path depth)
    # Middle:          layer N  = disease endpoint
    # Top section:     layer N+1 = direct genes
    #                  layer N+2 = direct variants

    _G_ep = nx.DiGraph()
    for _s, _t in edge_set:
        _G_ep.add_edge(_s, _t)

    # Classify genes: direct-only if every variant edge leads straight to endpoint
    _direct_genes: set[str] = set()
    for _prot in protein_nodes:
        _my_vars = {vid for vid in variant_nodes
                    if variant_nodes[vid]["protein"] == _prot}
        _out = {tgt for (src, tgt) in edge_set if src in _my_vars}
        if _out and _out <= {endpoint}:
            _direct_genes.add(_prot)
    _indirect_genes = set(protein_nodes.keys()) - _direct_genes

    # Nodes whose layer is set from the start (will be excluded from
    # the condensation-DAG propagation for intermediates)
    _direct_nodes: set[str] = _direct_genes | {
        vid for vid in variant_nodes
        if variant_nodes[vid]["protein"] in _direct_genes
    }

    node_layer: dict[str, int] = {}
    for _prot in _indirect_genes:
        node_layer[_prot] = 0
    for _vid in variant_nodes:
        if variant_nodes[_vid]["protein"] not in _direct_genes:
            node_layer[_vid] = 1

    # Longest-path depth for intermediate nodes (cycle-safe via condensation)
    _non_fixed = [n for n in _G_ep.nodes()
                  if n not in node_layer and n not in _direct_nodes and n != endpoint]
    if _non_fixed:
        _H = nx.DiGraph()
        _H.add_nodes_from(_non_fixed)
        for _u, _v in _G_ep.edges():
            if _u in _H and _v in _H:
                _H.add_edge(_u, _v)
        _init: dict[str, int] = {}
        for n in _non_fixed:
            _kp = [p for p in _G_ep.predecessors(n) if p in node_layer]
            _init[n] = (max(node_layer[p] for p in _kp) + 1) if _kp else 2
        _C = nx.condensation(_H)
        _scc_map = _C.graph["mapping"]
        _scc_d: dict[int, int] = {}
        for n in _non_fixed:
            sid = _scc_map[n]
            _scc_d[sid] = max(_scc_d.get(sid, 0), _init.get(n, 2))
        for sid in nx.topological_sort(_C):
            for succ_sid in _C.successors(sid):
                if _scc_d.get(succ_sid, 0) <= _scc_d.get(sid, 0):
                    _scc_d[succ_sid] = _scc_d[sid] + 1
        for n in _non_fixed:
            node_layer[n] = _scc_d[_scc_map[n]]

    # Disease endpoint sits above all indirect/intermediate layers
    _ep_layer = max(node_layer.values(), default=1) + 1
    node_layer[endpoint] = _ep_layer

    # Direct paths placed above disease; variant layer is closer to disease,
    # gene layer is one step further – mirroring the indirect section structure.
    for _prot in _direct_genes:
        node_layer[_prot] = _ep_layer + 2
    for _vid in variant_nodes:
        if variant_nodes[_vid]["protein"] in _direct_genes:
            node_layer[_vid] = _ep_layer + 1

    # --- Build layer groups with crossing-minimising orderings -----------
    _layers: dict[int, list] = defaultdict(list)
    for n, li in node_layer.items():
        _layers[li].append(n)

    def _order_gene_variant_layers(gene_layer: int, var_layer: int,
                                   genes: set, var_rank_source: list):
        """Sort variants grouped by gene; sort genes by variant centroid."""
        vl = sorted(
            _layers.get(var_layer, []),
            key=lambda vid: (
                variant_nodes[vid]["protein"].casefold() if vid in variant_nodes else "",
                vid.casefold(),
            ),
        )
        _layers[var_layer] = vl
        vrank = {vid: i for i, vid in enumerate(vl)}
        def _centroid(g):
            mv = [v for v in vl if v in variant_nodes and variant_nodes[v]["protein"] == g]
            return (sum(vrank[v] for v in mv) / len(mv)) if mv else 0.0
        _layers[gene_layer] = sorted(_layers.get(gene_layer, []), key=_centroid)
        return vl

    _indirect_var_list = _order_gene_variant_layers(0, 1, _indirect_genes, [])
    _direct_var_list   = _order_gene_variant_layers(
        _ep_layer + 2, _ep_layer + 1, _direct_genes, []
    )

    # Intermediate and endpoint layers: alphabetical baseline
    for li in _layers:
        if li not in {0, 1, _ep_layer + 1, _ep_layer + 2}:
            _layers[li].sort(key=lambda n: n.casefold())

    # LNS crossing optimisation – fix all gene/variant layers, optimise the rest
    _fixed_layers = {0, 1}
    if _direct_genes:
        _fixed_layers |= {_ep_layer + 1, _ep_layer + 2}

    _cross_w: dict[tuple, int] = {}
    for _s, _t in edge_set:
        ls, lt = node_layer.get(_s, 0), node_layer.get(_t, 0)
        if ls == lt:
            continue
        pair = (_s, _t) if ls < lt else (_t, _s)
        _cross_w[pair] = _cross_w.get(pair, 0) + 1
    _optimized = _optimize_layer_ordering(
        dict(_layers),
        [(u, v, w) for (u, v), w in _cross_w.items()],
        fixed_layers=_fixed_layers,
    )

    _layer_gap = 190.0
    _sp = 210.0
    pos: dict[str, tuple[float, float]] = {}
    for li, ordered in _optimized.items():
        n_nodes = len(ordered)
        start = -((n_nodes - 1) * _sp) / 2.0
        y = li * _layer_gap
        for i, n in enumerate(ordered):
            pos[n] = (start + i * _sp, y)

    # Align gene x to exact centroid of its variants' x-positions (both sections)
    for _var_list_section in (_indirect_var_list, _direct_var_list):
        for prot in protein_nodes:
            mv = [v for v in _var_list_section
                  if v in variant_nodes and variant_nodes[v]["protein"] == prot]
            xs = [pos[v][0] for v in mv if v in pos]
            if xs:
                pos[prot] = (sum(xs) / len(xs), pos.get(prot, (0, 0))[1])

    # --- Barycentric x for intermediate layers ----------------------------
    # Sparse intermediate layers tend to cluster in the centre; instead,
    # position each node at the average x of its lower-layer neighbours,
    # bounded by the span of the widest layer.
    _max_half = max(
        (len(nodes) - 1) * _sp / 2.0
        for nodes in _optimized.values()
        if len(nodes) > 1
    ) if any(len(v) > 1 for v in _optimized.values()) else _sp

    _adj_lo: dict[str, list] = defaultdict(list)
    for _s, _t in edge_set:
        if node_layer.get(_s, 0) < node_layer.get(_t, 0):
            _adj_lo[_t].append(_s)

    _fixed_li = {0, 1, _ep_layer, _ep_layer + 1, _ep_layer + 2}
    _inter_layers = sorted(li for li in _optimized if li not in _fixed_li)
    _min_sep = 80.0

    for li in _inter_layers:
        ordered = list(_optimized[li])
        if not ordered:
            continue
        y = li * _layer_gap

        # Barycentric x from lower-layer neighbours; fall back to current x
        bary: dict[str, float] = {}
        for nd in ordered:
            nbrs = [nb for nb in _adj_lo.get(nd, []) if nb in pos]
            bary[nd] = (
                sum(pos[nb][0] for nb in nbrs) / len(nbrs)
                if nbrs else pos.get(nd, (0.0, 0.0))[0]
            )

        # Sort by barycentric x, then enforce minimum spacing within bounds
        sorted_nodes = sorted(ordered, key=lambda nd: bary[nd])
        xs = [max(-_max_half, min(_max_half, bary[nd])) for nd in sorted_nodes]

        # Forward pass: minimum separation
        for idx in range(1, len(xs)):
            if xs[idx] < xs[idx - 1] + _min_sep:
                xs[idx] = xs[idx - 1] + _min_sep
        # Shift left if right boundary exceeded
        if xs and xs[-1] > _max_half:
            shift = xs[-1] - _max_half
            xs = [x - shift for x in xs]
        # Backward pass: fix any left-side violations after shift
        for idx in range(len(xs) - 2, -1, -1):
            if xs[idx] > xs[idx + 1] - _min_sep:
                xs[idx] = xs[idx + 1] - _min_sep

        for nd, x in zip(sorted_nodes, xs):
            pos[nd] = (x, y)

    def _ep_el(nid, data, role_cls, size):
        li = node_layer.get(nid, 0)
        el = {
            "data": data,
            "classes": f"L{li} {role_cls}",
            "style": {"width": size, "height": size},
        }
        if nid in pos:
            el["position"] = {"x": pos[nid][0], "y": pos[nid][1]}
        return el

    els = [_ep_el(endpoint, {
        "id": endpoint,
        "label": _short_label(endpoint, max_len=36),
        "real": endpoint,
        "role": "endpoint",
        "n_proteins": total_proteins,
        "n_variants": total_variants,
        "n_records": total_records,
    }, "role-endpoint", 110)]

    for prot, info in ordered_proteins:
        size = 48 + min(26, 5 * math.sqrt(max(len(info["variant_ids"]), 1)))
        els.append(_ep_el(prot, {
            "id": prot,
            "label": prot,
            "real": prot,
            "role": "protein",
            "n_variants": len(info["variant_ids"]),
            "n_records": info["row_count"],
            "protein_page": _protein_href(prot),
        }, "role-protein", size))

    for variant_id, info in ordered_variants:
        size = 42 + min(18, 4 * math.sqrt(max(info["row_count"], 1)))
        node_data = {
            "id": variant_id,
            "label": info["label"],
            "real": info["real"],
            "role": "variant",
            "gene_symbol": info["protein"],
            "n_records": info["row_count"],
            "protein_page": _protein_href(info["protein"]),
        }
        if info["allele_id"]:
            node_data["clinvar_allele"] = info["allele_id"]
        if info["dbsnp_rs"]:
            node_data["dbsnp_rs"] = info["dbsnp_rs"]
        if info["clinvar_data"]:
            node_data["clinvar_data"] = info["clinvar_data"]
        els.append(_ep_el(variant_id, node_data, "role-variant", size))

    for nid, info in sorted(intermediate_nodes.items(), key=lambda kv: kv[0].casefold()):
        size = 44 + min(20, 3 * math.sqrt(max(info["row_count"], 1)))
        node_data = {
            "id": nid,
            "label": info["label"],
            "real": info["real"],
            "role": "intermediate",
            "n_records": info["row_count"],
        }
        if info["is_known_protein"]:
            node_data["protein_page"] = _protein_href(info["real"])
        els.append(_ep_el(nid, node_data, "role-intermediate", size))

    for (src, tgt, rel), payload in sorted(edge_bucket.items()):
        if rel == "PV":
            cls = "edge-PV"
        else:
            cls = f"edge-{_css_safe(rel)}"
        edge_data = {
            "id": f"{src}->{tgt}::{rel}",
            "source": src,
            "target": tgt,
            "rel": rel,
            "src4indra": src,
            "evidence_count": payload["count"],
        }
        if payload["pmids"]:
            pmids = sorted(payload["pmids"])
            edge_data["pmid"] = "; ".join(pmids[:20])
        els.append({
            "data": edge_data,
            "classes": cls,
        })

    legend_rels = ["Gene–variant"] + raw_rel_types
    legend_colors = {
        "Gene–variant": "#c9c4bf",
        **{rel_display.get(k, k): v for k, v in rel_color_safe.items()},
    }
    return els, legend_rels, legend_colors, rel_color_safe, list(edge_set), {}


# --- Shared UI theme (warm paper / ink; avoids generic “AI gradient” look) ---
U = {
    "font_ui": (
        "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
        "Arial, sans-serif"
    ),
    "font_display": "Georgia, Cambria, 'Times New Roman', Times, serif",
    "ink": "#1c1b18",
    "ink_soft": "#454036",
    "muted": "#6f6b63",
    "paper": "#f2efe6",
    "panel": "#fdfbf7",
    "card": "#fffcf7",
    "wash": "#ebe6dc",
    "rule": "#e0dbd2",
    "link": "#2a4a66",
    "hero": "#262422",
    "hero_hi": "#32302c",
    "hero_deep": "#1a1917",
    "hero_text": "#f7f4ec",
    "hero_muted": "#c4beb4",
    "shadow": "0 1px 2px rgba(28, 27, 24, 0.05), 0 6px 20px rgba(28, 27, 24, 0.06)",
    "shadow_strong": "0 2px 8px rgba(28, 27, 24, 0.1)",
    "chart_bp": "#4d5f52",
    "chart_dis": "#8b534c",
    "chart_gene": "#4d5866",
    "plot_bp": "#eef1ec",
    "plot_dis": "#f3eceb",
    "plot_gene": "#eceff2",
    "accent_card_bp": "#6d7f72",
    "accent_card_dis": "#a1665f",
    "accent_card_gene": "#6a7484",
    "graph_bg": "#f7f5f0",
    "legend_bg": "rgba(253, 251, 247, 0.96)",
}

GRAPH_PROTEIN_BG = "#c5d2ce"
GRAPH_PROTEIN_FG = "#2a3d38"
GRAPH_VARIANT_BG = "#c9c0d4"
GRAPH_VARIANT_FG = "#3d324d"
GRAPH_INTERMEDIATE_BG = "#cfd9c3"
GRAPH_INTERMEDIATE_FG = "#35422e"
GRAPH_ENDPOINT_BG = "#e8d4bc"
GRAPH_ENDPOINT_FG = "#5c3f24"

# Cytoscape main graph `node` uses this opacity over `backgroundColor` (U["paper"]).
GRAPH_NODE_BG_OPACITY = 0.92


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        return 0, 0, 0
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _graph_node_fill_hex(fg_hex: str, blend_on: Optional[str] = None) -> str:
    """Blend fg onto canvas (default paper, e.g. card for the variant strip)."""
    fg = _hex_to_rgb(fg_hex)
    bg = _hex_to_rgb(blend_on or U["paper"])
    a = GRAPH_NODE_BG_OPACITY
    return _rgb_to_hex(
        int(fg[0] * a + bg[0] * (1 - a)),
        int(fg[1] * a + bg[1] * (1 - a)),
        int(fg[2] * a + bg[2] * (1 - a)),
    )


def _network_frame_style() -> dict:
    """White functional panel like stats chart cards."""
    return {
        "background": U["card"],
        "border": f"1px solid {U['rule']}",
        "borderRadius": 4,
        "boxShadow": U["shadow"],
        "overflow": "hidden",
    }


def _variant_strip_wrap_style() -> dict:
    """Variant strip: white surface, no frame (per design), allow markers to paint fully."""
    return {
        "background": U["card"],
        "border": "none",
        "boxShadow": "none",
        "borderRadius": 4,
        "overflow": "visible",
    }


# Header row under URL bar (approx.) — remainder is graph + bottom strip
_NETWORK_HEADER_PX = 96
_GENE_STRIP_VH_PCT = 22
_GENE_STRIP_VH = f"{_GENE_STRIP_VH_PCT}vh"
_GENE_STRIP_GRAPH_INNER_H = f"calc({_GENE_STRIP_VH_PCT}vh - 26px)"


def _protein_lollipop_figure(prot: str) -> Optional[go.Figure]:
    """Lollipop plot: wide protein bar as backbone, variant discs above with stems.

    The protein bar spans the full paper width so it is always visible.  A
    protein-node circle sits at the right boundary (on secondary x2 axis) and
    appears as a cap at the bar's right end.  Variant markers float above the bar
    connected by thin stems, at their amino-acid positions on the primary axis.
    """
    if prot not in TSV_FILES:
        return None
    df = pd.read_csv(TSV_FILES[prot], sep="\t").fillna("")
    pos_to_vars: dict[int, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        var = str(row.get("variant_info", "")).strip()
        if not var:
            continue
        pos = _variant_aa_position(var, str(row.get("Name", "")).strip())
        if pos is None:
            continue
        pos_to_vars[pos].add(var)
    if not pos_to_vars:
        return None

    ordered = {p: sorted(pos_to_vars[p], key=_sort_text) for p in sorted(pos_to_vars)}
    max_pos = max(ordered)
    min_pos = min(ordered)

    # Layout constants
    y_track  = 0.0    # vertical centre of the protein bar
    bar_half = 0.20   # half-height of the bar (thick backbone)
    y_var    = 0.72   # vertical centre of variant discs (above bar)

    vx, vy, vtxt, vhov = [], [], [], []
    for pos, labs in ordered.items():
        n = len(labs)
        xs = ([float(pos)] if n == 1 else
              [float(pos) + 0.45 * (i - (n - 1) / 2.0) for i in range(n)])
        for x, lab in zip(xs, labs):
            vx.append(x)
            vy.append(y_var)
            vtxt.append(lab)
            vhov.append(f"{lab} · aa {pos}<extra></extra>")

    fig = go.Figure()
    fill_protein = _graph_node_fill_hex(GRAPH_PROTEIN_BG, U["card"])
    fill_variant = _graph_node_fill_hex(GRAPH_VARIANT_BG, U["card"])

    # ── Protein backbone bar ──────────────────────────────────────────────────
    # Uses paper coordinates for x so it spans the full primary-axis area and
    # is never clipped when the user zooms. y is still in data space so the
    # bar aligns with variant stems correctly.
    # The bar extends to paper x=0.93 (the protein circle centre) so the bar
    # visually "connects" to and terminates at the protein cap node.
    fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0.0, x1=0.93,
        y0=y_track - bar_half, y1=y_track + bar_half,
        fillcolor=fill_protein, line=dict(width=0),
        layer="below",
    )

    # ── Stems: one vertical line per variant ─────────────────────────────────
    stem_x: list = []
    stem_y: list = []
    for px in vx:
        stem_x.extend([px, px, None])
        stem_y.extend([y_var, y_track + bar_half, None])
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color=fill_variant, width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Variant discs ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vx, y=vy, mode="markers+text",
        marker=dict(size=30, color=fill_variant, line=dict(width=0)),
        text=vtxt, textposition="middle center",
        textfont=dict(size=11, color=GRAPH_VARIANT_FG, family=U["font_ui"], weight=600),
        customdata=vhov, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    # ── Protein cap node (secondary x-axis, always fixed at right) ────────────
    # x2.domain=[0.87, 1.0], x2.range=[0,1], protein at x=0.5
    # → paper x = 0.87 + 0.5 * 0.13 = 0.935 ≈ 0.93.
    # The entire circle stays inside x2 domain so Plotly does not clip it.
    # The bar (paper x1=0.93) terminates at the circle centre giving an
    # "embedded cap" appearance identical to the reference image.
    fig.add_trace(go.Scatter(
        x=[0.5], y=[y_track],
        mode="markers+text",
        marker=dict(size=32, color=fill_protein, line=dict(width=0)),
        text=[prot], textposition="middle center",
        textfont=dict(size=12, color=GRAPH_PROTEIN_FG, family=U["font_ui"], weight=600),
        hovertemplate=f"<b>{prot}</b><extra></extra>",
        showlegend=False,
        xaxis="x2",
    ))

    # Left/right padding: proportional to protein length so variant circles
    # (radius ~13 px) are never clipped regardless of protein size.
    _pad = float(max_pos) * 0.03 + 15
    x_lo = -_pad
    x_hi = float(max_pos) * 1.06 + 30
    fig.update_layout(
        xaxis2=dict(
            range=[0, 1],          # x=0.5 → paper x = 0.87 + 0.5*0.13 = 0.935
            domain=[0.87, 1.0],    # wider x2 domain so circle is not clipped
            visible=False,
            fixedrange=True,
        ),
        margin=dict(l=54, r=14, t=16, b=30),
        plot_bgcolor=U["card"],
        paper_bgcolor=U["card"],
        font=dict(family=U["font_ui"], size=10, color=U["ink_soft"]),
        xaxis=dict(
            title="Amino acid position",
            range=[x_lo, x_hi],
            domain=[0, 0.87],
            showgrid=True, gridcolor="rgba(45,42,36,0.06)",
            zeroline=False,
            tickfont=dict(size=9, color=U["ink_soft"], family=U["font_ui"]),
            fixedrange=False,
        ),
        yaxis=dict(visible=False, range=[-0.52, 1.6], fixedrange=True),
        height=116,
        hovermode="closest",
        dragmode="zoom",
    )
    return fig


def _relayout_xaxis_range(relayout: Optional[dict]) -> Optional[tuple[float, float]]:
    """Parse Plotly relayout dict for x-axis range; None means no zoom box / full span."""
    if not relayout:
        return None
    if relayout.get("xaxis.autorange") is True:
        return None
    k0, k1 = "xaxis.range[0]", "xaxis.range[1]"
    if k0 not in relayout or k1 not in relayout:
        return None
    try:
        a = float(relayout[k0])
        b = float(relayout[k1])
    except (TypeError, ValueError):
        return None
    return (a, b) if a <= b else (b, a)


def _norm_map_range(mr) -> Optional[tuple[float, float]]:
    if not mr or not isinstance(mr, (list, tuple)) or len(mr) != 2:
        return None
    try:
        a, b = float(mr[0]), float(mr[1])
    except (TypeError, ValueError):
        return None
    return (a, b) if a <= b else (b, a)


def _cy_net_layout_preset() -> dict:
    return {"name": "preset"}


def _adjacency_from_elements(els: list) -> tuple[dict, dict]:
    """Directed forward and reverse adjacency from edge elements only."""
    fwd: dict = defaultdict(set)
    rev: dict = defaultdict(set)
    for el in els:
        d = el.get("data") or {}
        if "source" not in d:
            continue
        s, t = d["source"], d["target"]
        fwd[s].add(t)
        rev[t].add(s)
    return fwd, rev


# ------------------------Dash App------------------------–
app = dash.Dash(__name__,
                title="VarAtlas · INDRA Variant",
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.SANDSTONE])
# Set the server for deployment, see https://dash.plotly.com/deployment
server = app.server
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page")])


def _browse_panel(title: str, helper_text: str, dropdown_id: str, options: list,
                  placeholder: str, button_id: str, directory_id: str,
                  summary_text: str, note_text: Optional[str] = None):
    return html.Div([
        html.H4(title, style={'marginTop': 0, 'marginBottom': 10,
                              'color': U['ink'],
                              'fontFamily': U['font_display'],
                              'fontWeight': 600}),
        html.P(helper_text, style={'fontSize': 16, 'margin': '0 0 16px 0',
                                   'color': U['ink_soft'],
                                   'fontFamily': U['font_ui'],
                                   'lineHeight': 1.45}),
        dcc.Dropdown(id=dropdown_id, options=options,
                     placeholder=placeholder,
                     style={'fontSize': 16, 'fontFamily': U['font_ui']},
                     clearable=True,
                     searchable=True),
        dbc.Button("Open", id=button_id, n_clicks=0,
                   color="primary", style={'marginTop': 18,
                                           'fontSize': 16,
                                           'fontFamily': U['font_ui'],
                                           'fontWeight': 600,
                                           'padding': '8px 22px'}),
        html.Div(summary_text,
                 style={'marginTop': 16, 'marginBottom': 10,
                        'fontSize': 14, 'color': U['muted'],
                        'fontFamily': U['font_ui']}),
        *([] if not note_text else [
            html.Div(note_text,
                     style={'marginBottom': 14, 'fontSize': 14,
                            'color': U['muted'], 'fontStyle': 'italic',
                            'fontFamily': U['font_ui'], 'lineHeight': 1.45})
        ]),
        html.Div(id=directory_id,
                 style={'fontFamily': U['font_ui']})
    ], style={'padding': '12px 6px 6px'})


def _render_network_page(view_key: str, root_node_id: str, title: str,
                         graph_tuple: tuple, layout: Optional[dict] = None,
                         lollipop_figure: Optional[go.Figure] = None):
    els, legend_rels, legend_colors, rel_color_safe, edge_set, subgraph_data = graph_tuple
    layout = layout or {'name': 'preset'}

    def rel_style(css_cls, c):
        return {'selector': f'.edge-{css_cls}',
                'style': {'line-color': c, 'target-arrow-color': c,
                          'target-arrow-shape': 'triangle',
                          'curve-style': 'bezier', 'width': 2}}

    sidebar = html.Div(
        id={'type': 'edge-info', 'prot': view_key},
        children=[
            html.Div("Details",
                    style={'fontSize': 17, 'fontWeight': 600,
                           'fontFamily': U['font_display'],
                           'marginBottom': 15, 'color': U['ink'],
                           'borderBottom': f'1px solid {U["rule"]}',
                           'paddingBottom': 10}),
            html.Div("Select a node or edge for attributes, evidence, and external links.",
                    style={'color': U['muted'], 'fontSize': 14,
                           'lineHeight': '1.45', 'fontFamily': U['font_ui']})
        ],
        style={
            'position': 'fixed',
            'left': 0,
            'top': 0,
            'width': 350,
            'height': '100vh',
            'background': U['graph_bg'],
            'padding': '22px 20px',
            'boxShadow': U['shadow_strong'],
            'borderRight': f'1px solid {U["rule"]}',
            'fontSize': 15,
            'fontFamily': U['font_ui'],
            'zIndex': 1000,
            'overflowY': 'auto'
        }
    )

    main_content = html.Div([
        html.Div([
            dcc.Link("← Overview", href="/",
                    style={'color': U['link'], 'textDecoration': 'none',
                           'fontSize': 14, 'fontWeight': 600,
                           'fontFamily': U['font_ui']}),
            html.H4(title,
                   style={'textAlign': 'center', 'margin': '6px 0 2px',
                          'color': U['ink'],
                          'fontFamily': U['font_display'],
                          'fontWeight': 600}),
            html.P("Select the root node to clear highlighting.",
                   style={'textAlign': 'center', 'marginTop': 0,
                          'marginBottom': 15, 'color': U['muted'],
                          'fontFamily': U['font_ui'], 'fontSize': 13})
        ], style={'padding': '14px 16px', 'background': U['panel'],
                  'borderBottom': f'1px solid {U["rule"]}'}),

        dcc.Store(id={'type': 'store-els',  'prot': view_key},  data=els),
        dcc.Store(id={'type': 'store-edges', 'prot': view_key},  data=edge_set),
        dcc.Store(id={'type': 'store-root', 'prot': view_key},  data=root_node_id),
        dcc.Store(id={'type': 'store-subgraphs', 'prot': view_key}, data=subgraph_data),
        dcc.Store(id={'type': 'store-relcolors', 'prot': view_key}, data=rel_color_safe),
        dcc.Store(id={'type': 'store-map-range', 'prot': view_key}, data=None),
        *([] if lollipop_figure is not None else [
            html.Div([
                dcc.Graph(
                    id={'type': 'gene-map', 'prot': view_key},
                    figure={'data': [], 'layout': {'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0}}},
                    config={'displayModeBar': False, 'staticPlot': True},
                    style={'display': 'none', 'height': 0, 'width': 0},
                ),
                html.Button(
                    id={'type': 'gene-map-reset', 'prot': view_key},
                    n_clicks=0,
                    style={'display': 'none'},
                ),
            ], style={'display': 'none'}),
        ]),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                id={'type': 'subgraph-title', 'prot': view_key})),
            dbc.ModalBody(
                html.Div([
                    cyto.Cytoscape(
                        id={'type': 'cy-subgraph', 'prot': view_key},
                        elements=[], layout={'name': 'preset'},
                        style={'width': '70%', 'height': '100%',
                               'backgroundColor': U['paper']},
                        stylesheet=[]),
                    html.Div(
                        id={'type': 'subgraph-edge-info', 'prot': view_key},
                        children=[
                            html.Div("Select an edge for details.",
                                     style={'color': U['muted'], 'fontSize': 14,
                                            'padding': 16,
                                            'fontFamily': U['font_ui']})
                        ],
                        style={'width': '30%', 'height': '100%',
                               'overflowY': 'auto',
                               'borderLeft': f'1px solid {U["rule"]}',
                               'background': U['paper']})
                ], style={'display': 'flex', 'height': '70vh'}),
                style={'padding': 0}),
        ], id={'type': 'subgraph-modal', 'prot': view_key},
           size="xl", is_open=False),

        html.Div(
            [
                html.Div(
                    [
                        cyto.Cytoscape(
                            id={'type': 'cy-net', 'prot': view_key},
                            elements=els,
                            layout=layout,
                            style={
                                'width': '100%',
                                'height': '100%',
                                'flex': '1 1 auto',
                                'minHeight': 0,
                                'backgroundColor': U['card'],
                            },
                            stylesheet=[
                {'selector': 'node', 'style': {
                    'shape': 'ellipse', 'background-opacity': 0.92,
                    'font-size': 15, 'font-weight': '600',
                    'label': 'data(label)',
                    'text-wrap': 'wrap',
                    'text-max-width': 100,
                    'text-valign': 'center',
                    'text-halign': 'center'}},
                {'selector': '.role-protein',
                 'style': {'background-color': GRAPH_PROTEIN_BG,
                           'color': GRAPH_PROTEIN_FG}},
                {'selector': '.role-variant',
                 'style': {'background-color': GRAPH_VARIANT_BG,
                           'color': GRAPH_VARIANT_FG}},
                {'selector': '.role-intermediate',
                 'style': {'background-color': GRAPH_INTERMEDIATE_BG,
                           'color': GRAPH_INTERMEDIATE_FG}},
                {'selector': '.role-endpoint',
                 'style': {'background-color': GRAPH_ENDPOINT_BG,
                           'color': GRAPH_ENDPOINT_FG}},
                {'selector': '.edge-PV',
                 'style': {'line-color': '#c9c4bf',
                           'target-arrow-shape': 'triangle',
                           'width': 2}},
                *[rel_style(css_cls, c) for css_cls, c in rel_color_safe.items()],
                {'selector': '.faded', 'style': {'opacity': 0.15}}
            ],
                        ),
                    ],
                    style={
                        **_network_frame_style(),
                        'display': 'flex',
                        'flexDirection': 'column',
                        'flex': '1 1 auto',
                        'minHeight': 0,
                    },
                ),
                *([] if lollipop_figure is None else [
                    html.Div(
                        [
                            html.Div([
                                html.Span(
                                    "Variant map",
                                    style={
                                        'fontSize': 12, 'color': U['ink_soft'],
                                        'fontFamily': U['font_ui'],
                                        'fontWeight': 600,
                                        'letterSpacing': '0.06em',
                                        'textTransform': 'uppercase',
                                        'flex': '1',
                                    }),
                                dbc.Button(
                                    "Reset view",
                                    id={'type': 'gene-map-reset', 'prot': view_key},
                                    n_clicks=0, size="sm", outline=True,
                                    style={
                                        'fontSize': 11, 'padding': '2px 10px',
                                        'fontFamily': U['font_ui'],
                                    }),
                            ], style={
                                'display': 'flex', 'alignItems': 'center',
                                'justifyContent': 'space-between',
                                'padding': '6px 10px 4px',
                                'background': U['card'],
                            }),
                            dcc.Graph(
                                id={'type': 'gene-map', 'prot': view_key},
                                figure=lollipop_figure,
                                config={
                                    'displayModeBar': False,
                                    'scrollZoom': True,
                                    'doubleClick': False,
                                },
                                style={
                                    'height': _GENE_STRIP_GRAPH_INNER_H,
                                    'width': '100%', 'margin': 0,
                                    'paddingLeft': 6, 'paddingRight': 6,
                                    'boxSizing': 'border-box',
                                },
                            ),
                        ],
                        style={
                            **_variant_strip_wrap_style(),
                            'flex': f'0 0 {_GENE_STRIP_VH}',
                            'minHeight': 0,
                            'display': 'flex',
                            'flexDirection': 'column',
                        },
                    ),
                ]),
            ],
            style={
                'display': 'flex', 'flexDirection': 'column',
                'flex': '1 1 auto', 'minHeight': 0,
                'height': f'calc(100vh - {_NETWORK_HEADER_PX}px)',
                'gap': 10,
                'padding': '12px 14px 14px',
                'boxSizing': 'border-box',
            },
        ),

        html.Div([
            html.H4("Edge types",
                    style={'margin': 0, 'fontSize': 14,
                           'fontWeight': 600,
                           'fontFamily': U['font_display'],
                           'color': U['ink'],
                           'letterSpacing': '0.02em'}),
            html.Ul([
                html.Li([html.Span('→',
                                  style={'color': legend_colors.get(r, '#c9c4bf'),
                                         'marginRight': 8,
                                         'fontSize': 14}), r],
                        style={'fontSize': 13, 'listStyle': 'none',
                               'margin': '6px 0',
                               'color': U['ink_soft'],
                               'fontFamily': U['font_ui']})
                for r in legend_rels
            ], style={'paddingLeft': 0, 'margin': '10px 0 0 0'})
        ], style={'position': 'absolute',
                  'top': _NETWORK_HEADER_PX + 47, 'right': 22,
                  'background': U['card'],
                  'padding': '14px 18px',
                  'borderRadius': 4,
                  'border': f'1px solid {U["rule"]}',
                  'boxShadow': U['shadow'],
                  'fontFamily': U['font_ui'],
                  'maxHeight': '60vh',
                  'overflowY': 'auto',
                  'zIndex': 10})

    ], style={
        'marginLeft': 350,
        'position': 'relative',
        'height': '100vh',
        'backgroundColor': U['wash'],
        'display': 'flex',
        'flexDirection': 'column',
    })

    return html.Div([sidebar, main_content])


def _app_footer():
    return html.Div(
        [
            html.Span("Maintained by the "),
            html.A("Gyori Lab", href="https://gyorilab.github.io",
                   target="_blank",
                   style={'color': U['link'], 'fontWeight': 500}),
            html.Span(", Northeastern University."),
            html.Br(),
            html.Span("Supported by DARPA ASKEM / ARPA-H BDF (HR00112220036).")
        ],
        style={'background': U['wash'],
               'padding': '16px 24px',
               'textAlign': 'center', 'fontSize': 13,
               'fontFamily': U['font_ui'],
               'color': U['muted'],
               'borderTop': f'1px solid {U["rule"]}'},
    )


# ---Search / browse (gene & endpoint pickers)---
def search_page():
    search_card = html.Div(
        [
            dcc.Link(
                "← Overview",
                href="/",
                style={'color': U['link'], 'textDecoration': 'none',
                       'fontSize': 14, 'fontWeight': 600,
                       'display': 'inline-block', 'marginBottom': 10,
                       'fontFamily': U['font_ui']}),
            html.H1("Variant network browser",
                    style={'marginTop': 0, 'marginBottom': 14,
                           'color': U['ink'],
                           'fontFamily': U['font_display'],
                           'fontWeight': 600,
                           'fontSize': '2rem',
                           'letterSpacing': '-0.01em'}),
            html.P([
                "protein-centric graphs are keyed by protein; phenotype-centric graphs by "
                "disease or pathway term. ",
                dcc.Link("Summary metrics",
                         href="/",
                         style={'color': U['link'], 'fontWeight': 600}),
                " are on the overview.",
            ], style={'fontSize': 16, 'margin': '0 0 22px 0', 'color': U['ink_soft'],
                      'fontFamily': U['font_ui'], 'lineHeight': 1.5}),
            dcc.Tabs([
                dcc.Tab(
                    label="Protein-centric",
                    children=[
                        _browse_panel(
                            title="Protein",
                            helper_text="Search or browse A–Z",
                            dropdown_id='prot-search',
                            options=PROT_OPTIONS,
                            placeholder="Protein symbol…",
                            button_id='submit-prot',
                            directory_id='prot-directory',
                            summary_text=f"{len(PROTS)} graphs available.",
                        )
                    ]
                ),
                dcc.Tab(
                    label="Phenotype-centric",
                    children=[
                        _browse_panel(
                            title="Phenotype",
                            helper_text="Search or browse A–Z",
                            dropdown_id='endpoint-search',
                            options=ENDPOINT_OPTIONS,
                            placeholder="Biological process or disease term…",
                            button_id='submit-endpoint',
                            directory_id='endpoint-directory',
                            summary_text=(
                                f"{len(ENDPOINTS)} indexed phenotypes nodes."
                            ),
                        )
                    ]
                )
            ])
        ],
        style={'maxWidth': 860, 'margin': '48px auto',
               'background': U['panel'], 'padding': '40px 48px',
               'borderRadius': 4,
               'boxShadow': U['shadow'],
               'fontFamily': U['font_ui'],
               'border': f'1px solid {U["rule"]}'}
    )

    return html.Div([search_card, _app_footer()],
                    style={'minHeight': '100vh', 'background': U['paper']})


def statistics_page():
    """Landing page: static reference tables for top BPs / diseases / genes."""
    _cta_wrap = {
        'textDecoration': 'none',
        'display': 'block',
        'width': '100%',
    }
    hero = html.Div(
        [
            html.Div(
                style={'height': 3, 'background': '#b8a06e', 'opacity': 0.85}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "VarAtlas INDRA variant networks",
                                style={'fontSize': 11, 'letterSpacing': '0.14em',
                                       'textTransform': 'uppercase',
                                       'color': U['hero_muted'],
                                       'fontWeight': 600,
                                       'fontFamily': U['font_ui']}),
                            html.H1(
                                "Variant networks overview",
                                style={'fontSize': '2.05rem', 'fontWeight': 600,
                                       'fontFamily': U['font_display'],
                                       'color': U['hero_text'], 'margin': '12px 0 10px',
                                       'lineHeight': 1.2,
                                       'letterSpacing': '-0.02em'}),
                            html.P(
                                "Ranked biological processes, diseases, and genes "
                                "from the reference path statistics (static summary).",
                                style={'fontSize': 15, 'color': U['hero_muted'],
                                       'maxWidth': 620, 'margin': 0, 'lineHeight': 1.55,
                                       'fontFamily': U['font_ui']}),
                        ],
                        style={'flex': '1 1 320px', 'minWidth': 0}),
                    html.Div(
                        [
                            html.Div("Graphs", style={
                                'fontSize': 11, 'textTransform': 'uppercase',
                                'letterSpacing': '0.12em', 'color': U['hero_muted'],
                                'marginBottom': 10, 'fontWeight': 600,
                                'fontFamily': U['font_ui']}),
                            dcc.Link(
                                dbc.Button(
                                    "Open browser",
                                    className="w-100 mb-2",
                                    style={
                                        'fontWeight': 600,
                                        'fontFamily': U['font_ui'],
                                        'background': '#b8a06e',
                                        'color': '#1c1b18',
                                        'border': 'none',
                                        'letterSpacing': '0.02em',
                                        'boxShadow': '0 2px 8px rgba(184,160,110,0.35)',
                                    }),
                                href="/search",
                                style=_cta_wrap),
                            dcc.Link(
                                dbc.Button(
                                    "Protein-centric",
                                    className="w-100 mb-2",
                                    style={
                                        'fontWeight': 500,
                                        'fontFamily': U['font_ui'],
                                        'background': '#4d7c8a',
                                        'color': '#f7f4ec',
                                        'border': 'none',
                                        'letterSpacing': '0.02em',
                                    }),
                                href="/search",
                                style=_cta_wrap),
                            dcc.Link(
                                dbc.Button(
                                    "Phenotype-centric",
                                    className="w-100 mb-2",
                                    style={
                                        'fontWeight': 500,
                                        'fontFamily': U['font_ui'],
                                        'background': '#a1665f',
                                        'color': '#f7f4ec',
                                        'border': 'none',
                                        'letterSpacing': '0.02em',
                                    }),
                                href="/search",
                                style=_cta_wrap),
                            html.P(
                                "Opens /search — use tabs to switch view.",
                                style={'fontSize': 12, 'color': U['hero_muted'],
                                       'marginTop': 12, 'marginBottom': 0,
                                       'lineHeight': 1.45,
                                       'fontFamily': U['font_ui']}),
                        ],
                        style={'flex': '0 0 260px', 'maxWidth': '100%'}),
                ],
                style={'display': 'flex', 'flexWrap': 'wrap',
                       'gap': '32px 40px', 'alignItems': 'flex-start',
                       'justifyContent': 'space-between',
                       'padding': '32px 40px 36px'},
            ),
        ],
        style={
            'background': f'linear-gradient(165deg, {U["hero_hi"]} 0%, {U["hero"]} 55%, {U["hero_deep"]} 100%)',
            'borderBottom': f'1px solid {U["rule"]}',
            'boxShadow': U['shadow_strong'],
        },
    )

    metric_row = html.Div(
        [
            html.Span(
                "Metric",
                style={'fontWeight': 600, 'marginRight': 14, 'alignSelf': 'center',
                       'color': U['ink_soft'], 'fontSize': 13,
                       'fontFamily': U['font_ui'],
                       'letterSpacing': '0.04em', 'textTransform': 'uppercase'}),
            dbc.ButtonGroup(
                [
                    dbc.Button("Paths", id="stats-btn-path", n_clicks=0,
                               color="primary", outline=False,
                               style={'fontFamily': U['font_ui']}),
                    dbc.Button("Genes", id="stats-btn-gene", n_clicks=0,
                               color="secondary", outline=True,
                               style={'fontFamily': U['font_ui']}),
                    dbc.Button("Variants", id="stats-btn-variant", n_clicks=0,
                               color="secondary", outline=True,
                               style={'fontFamily': U['font_ui']}),
                    dbc.Button("PMIDs", id="stats-btn-pmid", n_clicks=0,
                               color="secondary", outline=True,
                               style={'fontFamily': U['font_ui']}),
                ],
                size="md",
            ),
            html.Span(
                "Same selection for all charts.",
                style={'marginLeft': 14, 'color': U['muted'], 'fontSize': 13,
                       'alignSelf': 'center', 'fontFamily': U['font_ui']}),
        ],
        style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center',
               'marginBottom': 8, 'padding': '14px 18px',
               'background': U['panel'],
               'borderRadius': 4,
               'border': f'1px solid {U["rule"]}',
               'boxShadow': U['shadow'],
               'width': '100%', 'boxSizing': 'border-box'},
    )

    explain = html.P(
        "Bars sort by the active metric (descending). "
        "Genes: gene count on process/disease charts; BP/disease term count on the gene chart. "
        "Click a bar to open its network when the label is indexed in this deployment.",
        style={'fontSize': 14, 'color': U['muted'], 'maxWidth': 920,
               'margin': '12px 0 0', 'lineHeight': 1.55,
               'fontFamily': U['font_ui']},
    )

    _card = lambda gid, accent, body_pad, fig_h: dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                id=gid,
                config={'displayModeBar': False},
                style={'height': fig_h, 'width': '100%'},
            ),
            style={'paddingTop': body_pad, 'paddingBottom': 12},
        ),
        className="mb-4",
        style={
            'width': '100%',
            'boxSizing': 'border-box',
            'border': f'1px solid {U["rule"]}',
            'borderRadius': 4,
            'overflow': 'hidden',
            'boxShadow': U['shadow'],
            'borderTop': f'3px solid {accent}',
            'background': U['card'],
        },
    )

    charts = html.Div(
        [
            _card("stats-fig-bp", U['accent_card_bp'], 4, STATS_FIG_HEIGHT_10),
            _card("stats-fig-disease", U['accent_card_dis'], 4, STATS_FIG_HEIGHT_10),
            _card("stats-fig-genes", U['accent_card_gene'], 4, STATS_FIG_HEIGHT_20),
        ],
        style={
            'marginTop': 22,
            'display': 'flex',
            'flexDirection': 'column',
            'gap': 16,
            'alignItems': 'stretch',
            'width': '100%',
            'maxWidth': 920,
        },
    )

    bottom_cta = html.Div(
        [
            html.Hr(style={'border': 'none', 'borderTop': f'1px solid {U["rule"]}',
                           'margin': '32px 0 22px'}),
            html.Div(
                [
                    html.Span(
                        "Open a gene or endpoint graph",
                        style={'fontSize': 15, 'color': U['ink_soft'],
                               'fontWeight': 600, 'marginRight': 16,
                               'fontFamily': U['font_display']}),
                    dcc.Link(
                        dbc.Button(
                            "Browser →",
                            color="primary",
                            style={'fontWeight': 600,
                                   'fontFamily': U['font_ui'],
                                   'padding': '8px 20px'}),
                        href="/search",
                        style={'textDecoration': 'none'}),
                ],
                style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center',
                       'gap': 12, 'justifyContent': 'center'},
            ),
        ],
    )

    body = html.Div(
        [
            hero,
            html.Div(
                [metric_row, explain, charts, bottom_cta, dcc.Store(id="stats-metric", data="path")],
                style={'maxWidth': 920, 'margin': '0 auto', 'padding': '28px 22px 48px',
                       'fontFamily': U['font_ui']},
            ),
            _app_footer(),
        ],
        style={'minHeight': '100vh', 'background': U['paper']},
    )
    return body


# ---Network Page---
def network_page(prot: str):
    return _render_network_page(
        view_key=f"protein::{prot}",
        root_node_id=prot,
        title=f"{prot} — variant network",
        graph_tuple=build_elements(prot),
        lollipop_figure=_protein_lollipop_figure(prot),
    )


def endpoint_network_page(endpoint: str):
    return _render_network_page(
        view_key=f"endpoint::{endpoint}",
        root_node_id=endpoint,
        title=f"{endpoint} — phenotypecentric network",
        graph_tuple=build_endpoint_elements(endpoint),
        layout={'name': 'preset'},
    )


# ----
@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(path):
    if path in (None, "/", "/statistics"):
        return statistics_page()
    if path == "/search":
        return search_page()
    if path.startswith("/protein/"):
        prot = _decode_route_value(path.split("/protein/", 1)[1])
        if prot in PROTS:
            return network_page(prot)
    if path.startswith("/endpoint/"):
        endpoint = _decode_route_value(path.split("/endpoint/", 1)[1])
        if endpoint in ENDPOINT_INDEX:
            return endpoint_network_page(endpoint)
    return html.Div(
        [
            html.H3("404 Not found",
                    style={'color': U['ink'], 'fontFamily': U['font_display'],
                           'fontWeight': 600}),
            html.P(
                [
                    dcc.Link("Overview", href="/",
                             style={'marginRight': 16, 'color': U['link'],
                                    'fontWeight': 600}),
                    dcc.Link("Browser", href="/search",
                             style={'color': U['link'], 'fontWeight': 600}),
                ],
                style={'fontSize': 15, 'fontFamily': U['font_ui']},
            ),
        ],
        style={'maxWidth': 560, 'margin': '80px auto',
               'fontFamily': U['font_ui'], 'background': U['paper'],
               'minHeight': '100vh', 'padding': '0 20px'},
    )

def _sidebar_default():
    return [
        html.Div("Details",
                 style={'fontSize': 17, 'fontWeight': 600,
                        'fontFamily': U['font_display'],
                        'marginBottom': 15, 'color': U['ink'],
                        'borderBottom': f'1px solid {U["rule"]}',
                        'paddingBottom': 10}),
        html.Div("Select a node or edge for attributes, evidence, and external links.",
                 style={'color': U['muted'], 'fontSize': 14, 'lineHeight': '1.45',
                        'fontFamily': U['font_ui']})
    ]


def _sidebar_card(children):
    return html.Div(children, style={
        'background': U['card'],
        'padding': 12,
        'borderRadius': 4,
        'boxShadow': U['shadow'],
        'marginBottom': 15,
        'border': f'1px solid {U["rule"]}',
        'fontFamily': U['font_ui'],
    })


def _build_edge_info(edge):
    if not edge:
        return _sidebar_default()

    content = [
        html.Div("Edge",
                 style={'fontSize': 18, 'fontWeight': 'bold',
                        'marginBottom': 15, 'color': U['ink'],
                        'borderBottom': f'1px solid {U["rule"]}',
                        'paddingBottom': 10})
    ]

    rel = edge.get('rel', 'N/A')
    source = edge.get('source', 'N/A')
    target = edge.get('target', 'N/A')

    content.append(_sidebar_card([
        html.Div("Relation",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': U['ink_soft'], 'marginBottom': 5}),
        html.Div(f"{source} → {target}",
                 style={'fontSize': 14, 'color': U['ink'],
                        'marginBottom': 8, 'fontWeight': 'bold'}),
        *([] if rel in ['DV', 'PV', 'has_domain'] else [
            html.Div(f"Type: {rel}",
                     style={'fontSize': 14, 'color': U['muted']})
        ]),
        *([] if not edge.get('evidence_count') else [
            html.Div(f"Statements: {edge['evidence_count']}",
                     style={'fontSize': 13, 'color': U['muted'], 'marginTop': 4})
        ])
    ]))

    if rel == 'DV':
        if edge.get('note'):
            content.append(_sidebar_card([
                html.Div("Domain",
                         style={'fontSize': 14, 'fontWeight': 'bold',
                                'color': U['ink_soft'], 'marginBottom': 8}),
                html.Div(edge['note'],
                         style={'fontSize': 14, 'color': U['ink'],
                                'lineHeight': '1.4'})
            ]))
    else:
        if edge.get('note'):
            content.append(_sidebar_card([
                html.Div("Description",
                         style={'fontSize': 14, 'fontWeight': 'bold',
                                'color': U['ink_soft'], 'marginBottom': 8}),
                html.Div(edge['note'],
                         style={'fontSize': 14, 'color': U['ink'],
                                'lineHeight': '1.4'})
            ]))

    if edge.get('clinvar_data'):
        data = edge['clinvar_data']
        content.append(_sidebar_card([
            html.Div("ClinVar",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 10}),
            html.Div([
                html.Div([
                    html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('pathogenicity', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Review: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('review', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Condition: ", style={'fontWeight': 'bold'}),
                    html.Div(data.get('conditions', 'N/A'),
                             style={'marginTop': 4, 'fontStyle': 'italic'})
                ])
            ], style={'fontSize': 13, 'color': U['ink'], 'lineHeight': '1.4'})
        ]))

    if edge.get('pmid'):
        pmid_raw = edge['pmid']
        pmid_list = [p.strip() for p in pmid_raw.replace(";", ",").split(",") if p.strip()]
        pubmed_links = []
        for pid in pmid_list:
            pubmed_links.append(
                html.A(f"PMID {pid}",
                       href=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                       target="_blank",
                       style={'display': 'inline-block', 'marginRight': 10,
                              'marginBottom': 4, 'color': U['link'],
                              'textDecoration': 'none', 'fontSize': 13})
            )
        content.append(_sidebar_card([
            html.Div("Links",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 10}),
            html.Div(pubmed_links,
                     style={'marginBottom': 8, 'lineHeight': '1.8'}),
            html.A("INDRA Discovery",
                   href=(f"https://discovery.indra.bio/search/"
                         f"?agent={_url.quote_plus(edge['src4indra'])}"
                         f"&other_agent={_url.quote_plus(edge['target'])}"
                         "&agent_role=subject&other_role=object"),
                   target="_blank",
                   style={'display': 'block', 'color': U['link'],
                          'textDecoration': 'none', 'fontSize': 13,
                          'fontWeight': 'bold'})
        ]))

    return content


def _build_node_info(node):
    if not node:
        return _sidebar_default()

    real_name = node.get('real', node.get('label', 'N/A'))
    role = node.get('role', 'unknown').replace('-', ' ').replace('_', ' ').title()

    content = [
        html.Div("Node",
                 style={'fontSize': 18, 'fontWeight': 'bold',
                        'marginBottom': 15, 'color': U['ink'],
                        'borderBottom': f'1px solid {U["rule"]}',
                        'paddingBottom': 10})
    ]

    content.append(_sidebar_card([
        html.Div("Name",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': U['ink_soft'], 'marginBottom': 5}),
        html.Div(real_name,
                 style={'fontSize': 14, 'color': U['ink'],
                        'marginBottom': 8, 'fontWeight': 'bold',
                        'lineHeight': '1.4'}),
        html.Div(f"Role: {role}",
                 style={'fontSize': 13, 'color': U['muted']})
    ]))

    stat_lines = []
    if node.get('n_proteins') is not None:
        stat_lines.append(html.Div([
            html.Span("Proteins: ", style={'fontWeight': 'bold'}),
            html.Span(str(node['n_proteins']))
        ], style={'marginBottom': 6}))
    if node.get('n_variants') is not None:
        stat_lines.append(html.Div([
            html.Span("Variants: ", style={'fontWeight': 'bold'}),
            html.Span(str(node['n_variants']))
        ], style={'marginBottom': 6}))
    if node.get('n_records') is not None:
        stat_lines.append(html.Div([
            html.Span("Source rows: ", style={'fontWeight': 'bold'}),
            html.Span(str(node['n_records']))
        ]))
    if stat_lines:
        content.append(_sidebar_card([
            html.Div("Counts",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 10}),
            *stat_lines
        ]))

    if node.get('domain_notes'):
        content.append(_sidebar_card([
            html.Div("Domain notes",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 8}),
            html.Div(node['domain_notes'],
                     style={'fontSize': 14, 'color': U['ink'],
                            'lineHeight': '1.4'})
        ]))

    if node.get('clinvar_data'):
        data = node['clinvar_data']
        content.append(_sidebar_card([
            html.Div("ClinVar",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 10}),
            html.Div([
                html.Div([
                    html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('pathogenicity', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Review: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('review', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Condition: ", style={'fontWeight': 'bold'}),
                    html.Div(data.get('conditions', 'N/A'),
                             style={'marginTop': 4, 'fontStyle': 'italic'})
                ])
            ], style={'fontSize': 13, 'color': U['ink'], 'lineHeight': '1.4'})
        ]))

    external_links = [
        html.A("INDRA Discovery",
               href=f"https://discovery.indra.bio/search/?agent={_url.quote_plus(real_name)}",
               target="_blank",
               style={'display': 'block', 'color': U['link'],
                      'textDecoration': 'none', 'fontSize': 13,
                      'fontWeight': 'bold', 'marginBottom': 6})
    ]
    if node.get('protein_page'):
        external_links.insert(
            0,
            dcc.Link("Protein-centric graph",
                     href=node['protein_page'],
                     style={'display': 'block', 'color': U['link'],
                            'textDecoration': 'none', 'fontSize': 13,
                            'fontWeight': 'bold', 'marginBottom': 6})
        )
    if node.get('uniprot_id'):
        uid = node['uniprot_id']
        if '_' in uid:
            uniprot_href = (
                f"https://www.uniprot.org/uniprotkb/"
                f"{_url.quote_plus(uid)}/entry"
            )
        else:
            uniprot_href = (
                "https://www.uniprot.org/uniprotkb?query="
                f"{_url.quote_plus(f'gene_exact:{uid} AND organism_id:9606')}"
            )
        external_links.append(
            html.A("UniProt",
                   href=uniprot_href,
                   target="_blank",
                   style={'display': 'block', 'color': U['link'],
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    if node.get('clinvar_allele'):
        external_links.append(
            html.A("ClinVar",
                   href=("https://www.ncbi.nlm.nih.gov/clinvar/?term="
                         f"{_url.quote_plus(str(node['clinvar_allele']) + '[alleleid]')}"),
                   target="_blank",
                   style={'display': 'block', 'color': U['link'],
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    elif node.get('gene_symbol'):
        clinvar_query = f"{node['gene_symbol']}[gene] AND {real_name}"
        external_links.append(
            html.A("ClinVar (search)",
                   href=("https://www.ncbi.nlm.nih.gov/clinvar/?term="
                         f"{_url.quote_plus(clinvar_query)}"),
                   target="_blank",
                   style={'display': 'block', 'color': U['link'],
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    if node.get('dbsnp_rs'):
        external_links.append(
            html.A("dbSNP",
                   href=f"https://www.ncbi.nlm.nih.gov/snp/rs{node['dbsnp_rs']}",
                   target="_blank",
                   style={'display': 'block', 'color': U['link'],
                          'textDecoration': 'none', 'fontSize': 13})
        )

    content.append(_sidebar_card([
        html.Div("Links",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': U['ink_soft'], 'marginBottom': 10}),
        *external_links
    ]))

    return content


@app.callback(
    Output({'type': 'edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapNodeData'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_sidebar_info(node, edge):
    if not dash.ctx.triggered:
        return dash.no_update

    prop = dash.ctx.triggered[0]['prop_id'].split('.')[-1]
    if prop == 'tapNodeData' and node:
        return _build_node_info(node)
    if prop == 'tapEdgeData' and edge:
        return _build_edge_info(edge)
    return dash.no_update


@app.callback(
    Output({'type': 'gene-map', 'prot': MATCH}, 'figure'),
    Input({'type': 'gene-map-reset', 'prot': MATCH}, 'n_clicks'),
    State({'type': 'gene-map', 'prot': MATCH}, 'id'),
    prevent_initial_call=True)
def reset_gene_variant_map(_n_clicks, gid):
    rid = (gid or {}).get('prot', '')
    if not rid.startswith('protein::'):
        return dash.no_update
    gene = rid.split('::', 1)[1]
    fig = _protein_lollipop_figure(gene)
    return fig if fig is not None else dash.no_update


@app.callback(
    Output({'type': 'store-map-range', 'prot': MATCH}, 'data'),
    Input({'type': 'gene-map', 'prot': MATCH}, 'relayoutData'),
    Input({'type': 'gene-map-reset', 'prot': MATCH}, 'n_clicks'),
    prevent_initial_call=True)
def sync_gene_map_x_range(relayout, _n_reset):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get('type') == 'gene-map-reset':
        return None
    xr = _relayout_xaxis_range(relayout if isinstance(relayout, dict) else None)
    if xr is None:
        return dash.no_update
    return [xr[0], xr[1]]


# ---------------------- subgraph modal callback ----------------------------
def _css_safe_global(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '-', name)

@app.callback(
    [Output({'type': 'subgraph-modal', 'prot': MATCH}, 'is_open'),
     Output({'type': 'subgraph-title', 'prot': MATCH}, 'children'),
     Output({'type': 'cy-subgraph',    'prot': MATCH}, 'elements'),
     Output({'type': 'cy-subgraph',    'prot': MATCH}, 'stylesheet')],
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapNodeData'),
    [State({'type': 'store-subgraphs',  'prot': MATCH}, 'data'),
     State({'type': 'store-relcolors',  'prot': MATCH}, 'data')],
    prevent_initial_call=True)
def open_subgraph_modal(node, subgraphs, rel_colors):
    empty = (False, "", [], [])
    if not node or not subgraphs:
        return empty
    nid = node.get("id", "")
    if nid not in subgraphs:
        return empty

    sg = subgraphs[nid]
    parent_name = sg["parent_name"]
    nodes_map = sg["nodes"]
    edges_list = sg["edges"]
    rel_colors = rel_colors or {}

    role_to_layer = {"protein": 0, "variant": 0, "intermediate": 1, "endpoint": 2}
    layer_y = {0: 400, 1: 220, 2: 40}
    role_cls = {
        "protein": "role-protein", "variant": "role-variant",
        "intermediate": "role-intermediate", "endpoint": "role-endpoint",
    }
    role_size = {"protein": 70, "variant": 50, "intermediate": 46, "endpoint": 50}

    buckets = {0: [], 1: [], 2: []}
    for nid_key, info in nodes_map.items():
        ly = role_to_layer.get(info["role"], 2)
        buckets[ly].append(nid_key)

    x_pos = {}
    for ly, ids in buckets.items():
        spacing = 160
        start = -((len(ids) - 1) * spacing) / 2.0
        for i, n in enumerate(sorted(ids)):
            x_pos[n] = start + i * spacing

    sub_els = []
    n_endpoints = len(buckets[2])
    for nid_key, info in nodes_map.items():
        ly = role_to_layer.get(info["role"], 2)
        cls = role_cls.get(info["role"], "role-endpoint")
        sz = role_size.get(info["role"], 46)
        sub_els.append({
            "data": {"id": nid_key, "label": info["label"]},
            "position": {"x": x_pos.get(nid_key, 0), "y": layer_y[ly]},
            "classes": cls,
            "style": {"width": sz, "height": sz}
        })

    for e in edges_list:
        rel = e.get("rel", "")
        if rel == "PV":
            cls = "edge-PV"
        else:
            cls = f"edge-{_css_safe_global(rel)}"
        edge_data = {
            "source": e["source"], "target": e["target"],
            "rel": rel, "label": rel,
        }
        for k in ("pmid", "note", "clinvar_data", "evidence_count"):
            if k in e and e[k]:
                edge_data[k] = e[k]
        sub_els.append({"data": edge_data, "classes": cls})

    stylesheet = [
        {"selector": "node", "style": {
            "label": "data(label)",
            "text-valign": "center", "text-halign": "center",
            "font-size": 11, "font-weight": "bold",
            "background-opacity": GRAPH_NODE_BG_OPACITY,
            "text-wrap": "wrap", "text-max-width": 110}},
        {"selector": ".role-protein", "style": {
            "background-color": GRAPH_PROTEIN_BG, "color": GRAPH_PROTEIN_FG}},
        {"selector": ".role-variant", "style": {
            "background-color": GRAPH_VARIANT_BG, "color": GRAPH_VARIANT_FG}},
        {"selector": ".role-intermediate", "style": {
            "background-color": GRAPH_INTERMEDIATE_BG,
            "color": GRAPH_INTERMEDIATE_FG}},
        {"selector": ".role-endpoint", "style": {
            "background-color": GRAPH_ENDPOINT_BG,
            "color": GRAPH_ENDPOINT_FG}},
        {"selector": ".edge-PV", "style": {
            "line-color": "#c9c4bf", "target-arrow-color": "#c9c4bf",
            "target-arrow-shape": "triangle", "curve-style": "bezier",
            "width": 1.5, "label": "data(label)",
            "font-size": 9, "color": "#a8a29e", "text-rotation": "autorotate"}},
    ]
    for css_cls, color in rel_colors.items():
        stylesheet.append({
            "selector": f".edge-{css_cls}", "style": {
                "line-color": color, "target-arrow-color": color,
                "target-arrow-shape": "triangle", "curve-style": "bezier",
                "width": 2, "label": "data(label)",
                "font-size": 9, "color": color, "text-rotation": "autorotate"}
        })

    title = f"{parent_name} — {n_endpoints} endpoints"
    return True, title, sub_els, stylesheet

# ---------------------- subgraph edge-info callback ------------------------
@app.callback(
    Output({'type': 'subgraph-edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-subgraph', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_subgraph_edge_info(edge):
    if not edge:
        return [html.Div("Select an edge for details.",
                         style={'color': U['muted'], 'fontSize': 14, 'padding': 16})]

    rel = edge.get('rel', '')
    source = edge.get('source', '')
    target = edge.get('target', '')

    card_style = {'background': U['card'], 'padding': 12, 'borderRadius': 4,
                  'boxShadow': U['shadow'],
                  'marginBottom': 12, 'border': f'1px solid {U["rule"]}',
                  'fontFamily': U['font_ui']}

    content = [
        html.Div("Edge",
                 style={'fontSize': 16, 'fontWeight': 'bold', 'color': U['ink'],
                        'borderBottom': f'1px solid {U["rule"]}', 'paddingBottom': 8,
                        'marginBottom': 12, 'padding': '12px 16px 8px'}),
        html.Div([
            html.Div(f"{source} → {target}",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': U['ink'], 'marginBottom': 6}),
            *([] if rel in ('PV',) else [
                html.Div(f"Type: {rel}",
                         style={'fontSize': 13, 'color': U['muted']})]),
            *([] if not edge.get('evidence_count') else [
                html.Div(f"Statements: {edge['evidence_count']}",
                         style={'fontSize': 12, 'color': U['muted'], 'marginTop': 4})]),
        ], style={**card_style, 'margin': '0 12px 12px'}),
    ]

    if edge.get('note'):
        content.append(html.Div([
            html.Div("Description",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 6}),
            html.Div(edge['note'],
                     style={'fontSize': 13, 'color': U['ink'], 'lineHeight': '1.4'})
        ], style={**card_style, 'margin': '0 12px 12px'}))

    if edge.get('clinvar_data'):
        data = edge['clinvar_data']
        content.append(html.Div([
            html.Div("ClinVar",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 6}),
            html.Div([
                html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('pathogenicity', 'N/A'))
            ], style={'fontSize': 12, 'marginBottom': 4}),
            html.Div([
                html.Span("Review: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('review', ''))
            ], style={'fontSize': 12, 'marginBottom': 4}),
            html.Div([
                html.Span("Condition: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('conditions', ''))
            ], style={'fontSize': 12, 'lineHeight': '1.4'}),
        ], style={**card_style, 'margin': '0 12px 12px'}))

    if edge.get('pmid'):
        raw = str(edge['pmid'])
        pmids = [p.strip() for p in re.split(r'[;,]', raw) if p.strip()]
        links = []
        for p in pmids:
            links.append(html.A(
                f"PMID {p}",
                href=f"https://pubmed.ncbi.nlm.nih.gov/{p}/",
                target="_blank",
                style={'display': 'block', 'color': U['link'],
                       'textDecoration': 'none', 'fontSize': 12,
                       'marginBottom': 3}))
        content.append(html.Div([
            html.Div("PubMed",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': U['ink_soft'], 'marginBottom': 6}),
            *links
        ], style={**card_style, 'margin': '0 12px 12px'}))

    return content

# ---------------------- highlight callback ---------------------------------
@app.callback(
    [Output({'type': 'cy-net', 'prot': MATCH}, 'elements'),
     Output({'type': 'cy-net', 'prot': MATCH}, 'layout')],
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapNodeData'),
    Input({'type': 'store-map-range', 'prot': MATCH}, 'data'),
    [State({'type': 'store-els', 'prot': MATCH}, 'data'),
     State({'type': 'store-root', 'prot': MATCH}, 'data')],
    prevent_initial_call=True)
def highlight(node, map_range, elements, root_prot):
    base = copy.deepcopy(elements)
    xr = _norm_map_range(map_range)
    trig = ctx.triggered_id
    range_changed = (
        isinstance(trig, dict) and trig.get("type") == "store-map-range"
    )
    if xr and root_prot in TSV_FILES:
        els = build_elements(root_prot, variant_aa_range=xr)[0]
        # Always preset so node `position` from the same layered algorithm as the full graph is applied.
        layout_out = _cy_net_layout_preset()
    else:
        els = base
        layout_out = (
            _cy_net_layout_preset() if range_changed else dash.no_update
        )

    fwd, rev = _adjacency_from_elements(els)

    def _strip_faded():
        for el in els:
            c = el.get('classes') or ''
            el['classes'] = c.replace(' faded', '')

    if not node:
        _strip_faded()
        return els, layout_out

    if node['id'] == root_prot:
        _strip_faded()
        return els, layout_out

    sel = node['id']
    keep_nodes = {sel}
    keep_edges = set()

    stack = [sel]
    while stack:
        cur = stack.pop()
        for t in fwd.get(cur, ()):
            if (cur, t) not in keep_edges:
                keep_edges.add((cur, t))
                keep_nodes.add(t)
                stack.append(t)

    stack = [sel]
    while stack:
        cur = stack.pop()
        for s in rev.get(cur, ()):
            if (s, cur) not in keep_edges:
                keep_edges.add((s, cur))
                keep_nodes.add(s)
                stack.append(s)

    for el in els:
        d = el.get('data') or {}
        if 'source' in d:
            keep = ((d['source'], d['target']) in keep_edges
                    or d.get('rel') == 'PV')
        else:
            keep = d.get('id') in keep_nodes

        c = el.get('classes') or ''
        if keep:
            el['classes'] = c.replace(' faded', '')
        else:
            if 'faded' not in c:
                el['classes'] = c + ' faded'

    return els, layout_out


# ---------------------- Search -------------------------------
@app.callback(Output('prot-directory', 'children'),
              Input('prot-search', 'value'))
def filter_directory(query):
    return _build_alpha_directory(PROTS, query, _protein_href, columns=3)


@app.callback(Output('endpoint-directory', 'children'),
              Input('endpoint-search', 'value'))
def filter_endpoint_directory(query):
    return _build_alpha_directory(ENDPOINTS, query, _endpoint_href, columns=2)


@app.callback(Output('url', 'href', allow_duplicate=True),
              Input('submit-prot', 'n_clicks'),
              State('prot-search', 'value'),
              prevent_initial_call=True)
def jump_to_protein(_, value):
    return _protein_href(value) if value else dash.no_update


@app.callback(Output('url', 'href', allow_duplicate=True),
              Input('submit-endpoint', 'n_clicks'),
              State('endpoint-search', 'value'),
              prevent_initial_call=True)
def jump_to_endpoint(_, value):
    return _endpoint_href(value) if value else dash.no_update


@app.callback(
    Output("url", "href", allow_duplicate=True),
    Input("stats-fig-bp", "clickData"),
    Input("stats-fig-disease", "clickData"),
    Input("stats-fig-genes", "clickData"),
    prevent_initial_call=True,
)
def stats_bar_open_network(bp_click, dis_click, genes_click):
    trig = ctx.triggered_id
    if trig == "stats-fig-bp":
        payload = bp_click
    elif trig == "stats-fig-disease":
        payload = dis_click
    elif trig == "stats-fig-genes":
        payload = genes_click
    else:
        return dash.no_update
    if not payload or not payload.get("points"):
        return dash.no_update
    href = _stats_point_href(payload["points"][0])
    return href if href else dash.no_update


@app.callback(
    Output("stats-metric", "data"),
    Input("stats-btn-path", "n_clicks"),
    Input("stats-btn-gene", "n_clicks"),
    Input("stats-btn-variant", "n_clicks"),
    Input("stats-btn-pmid", "n_clicks"),
    prevent_initial_call=True,
)
def stats_set_metric(_p, _g, _v, _m):
    key = ctx.triggered_id
    return {
        "stats-btn-path": "path",
        "stats-btn-gene": "gene",
        "stats-btn-variant": "variant",
        "stats-btn-pmid": "pmid",
    }.get(key, "path")


@app.callback(
    Output("stats-fig-bp", "figure"),
    Output("stats-fig-disease", "figure"),
    Output("stats-fig-genes", "figure"),
    Output("stats-btn-path", "color"),
    Output("stats-btn-path", "outline"),
    Output("stats-btn-gene", "color"),
    Output("stats-btn-gene", "outline"),
    Output("stats-btn-variant", "color"),
    Output("stats-btn-variant", "outline"),
    Output("stats-btn-pmid", "color"),
    Output("stats-btn-pmid", "outline"),
    Input("stats-metric", "data"),
)
def stats_render(metric):
    metric = metric or "path"
    fig_bp = _stats_bar_figure(
        _STATS_BP_ROWS, metric,
        "Biological processes (top 10)",
        for_gene_chart=False,
        value_fn=_stats_value_bp_disease,
        bar_color=U["chart_bp"],
        plot_bg=U["plot_bp"],
        paper_bg="rgba(0,0,0,0)",
        height=STATS_FIG_HEIGHT_10,
    )
    fig_dis = _stats_bar_figure(
        _STATS_DISEASE_ROWS, metric,
        "Diseases (top 10)",
        for_gene_chart=False,
        value_fn=_stats_value_bp_disease,
        bar_color=U["chart_dis"],
        plot_bg=U["plot_dis"],
        paper_bg="rgba(0,0,0,0)",
        height=STATS_FIG_HEIGHT_10,
    )
    fig_genes = _stats_bar_figure(
        _STATS_GENE_ROWS, metric,
        "Genes (top 20)",
        for_gene_chart=True,
        value_fn=_stats_value_gene,
        bar_color=U["chart_gene"],
        plot_bg=U["plot_gene"],
        paper_bg="rgba(0,0,0,0)",
        height=STATS_FIG_HEIGHT_20,
    )
    btn = []
    for m in ("path", "gene", "variant", "pmid"):
        active = metric == m
        btn.append("primary" if active else "secondary")
        btn.append(not active)
    return (fig_bp, fig_dis, fig_genes, *btn)


# --------------------Run App----------------------------–
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
