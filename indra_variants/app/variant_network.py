import re
import math
import copy
import itertools as _it
import logging
import time as _time
import urllib.parse as _url
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from typing import Optional

from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
from dash import html, dcc, Input, Output, State, MATCH

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
# ---------- OBO ontology cache (pre-built JSON) ----------
import json as _json

_OBO_CACHE_PATH = Path(DATA_DIR).parent.parent / "obo_endpoint_cache.json"

if _OBO_CACHE_PATH.exists():
    with open(_OBO_CACHE_PATH) as _f:
        _OBO_CACHE = _json.load(_f)
else:
    _OBO_CACHE = {}


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
                               'color': '#0366d6',
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
    return html.Div("No matches found.",
                    style={'color': '#6c757d', 'fontSize': 16, 'padding': '12px 0'})


def _find_endpoint_groups(endpoint_names: set, min_group: int = 3,
                          max_depth: int = 2) -> list:
    """Group endpoint names by shared ontology parent (from pre-built cache).

    Returns list of (parent_name, [member_endpoint_names]).
    Each endpoint appears in at most one group (the tightest / shallowest).
    """
    parent_to_children = defaultdict(set)
    for name in endpoint_names:
        entry = _OBO_CACHE.get(name)
        if not entry:
            continue
        for anc_id, anc_info in entry.get("ancestors", {}).items():
            depth = anc_info.get("depth", 99)
            if depth <= max_depth:
                parent_to_children[(anc_id, depth, anc_info.get("name", anc_id))].add(name)

    candidates = sorted(parent_to_children.items(),
                        key=lambda kv: (kv[0][1], -len(kv[1])))

    used: set = set()
    groups = []
    for (_, _depth, parent_name), members in candidates:
        available = members - used
        if len(available) < min_group:
            continue
        groups.append((parent_name, sorted(available)))
        used |= available

    return groups

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
    
# ----------------------Build Graph--------------------------–
def build_elements(prot: str):
    df_path = TSV_FILES[prot]
    df = pd.read_csv(df_path, sep="\t").fillna('')

    def extract_protein_position(variant_label: str, name_label: str) -> Optional[int]:
        m = AA_SUB_RE.match((variant_label or "").strip())
        if m:
            return int(m.group("pos"))
        m = P_DOT_RE.search((name_label or "").strip())
        if m:
            return int(m.group("pos"))
        return None

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
        protein_pos = extract_protein_position(var, name_label)
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

    motif_kind = {}
    motif_members = {}
    subgraph_data = {}

    # ---------- Ontology-based endpoint aggregation ----------
    endpoint_groups = _find_endpoint_groups(endpoints, min_group=3, max_depth=2)
    for parent_name, members in endpoint_groups:
        gid = f"onto_group::{parent_name}"
        G.add_node(gid)
        motif_kind[gid] = "onto_group"
        motif_members[gid] = members

        sub_nodes = {}
        sub_edges_map = {}
        incoming = defaultdict(set)
        outgoing = defaultdict(set)
        members_set = set(members)

        def _edge_payload(d):
            p = {"rel": d.get("relation", "")}
            if d.get("pmid"):   p["pmid"] = d["pmid"]
            if d.get("note"):   p["note"] = d["note"]
            if d.get("clinvar_data"): p["clinvar_data"] = d["clinvar_data"]
            if d.get("weight"): p["evidence_count"] = d["weight"]
            return p

        def _role(n):
            if n == prot: return "protein"
            if n in variants: return "variant"
            return "intermediate"

        for m in members:
            if not G.has_node(m):
                continue
            sub_nodes[m] = {"label": m, "role": "endpoint"}

            for u, _, d in G.in_edges(m, data=True):
                if u in members_set:
                    continue
                incoming[u].add(d.get("relation", ""))
                role = _role(u)
                sub_nodes.setdefault(u, {"label": u, "role": role})
                ek = (u, m, d.get("relation", ""))
                if ek not in sub_edges_map:
                    sub_edges_map[ek] = _edge_payload(d)

                if role == "intermediate":
                    for uu, _, dd in G.in_edges(u, data=True):
                        r2 = _role(uu)
                        sub_nodes.setdefault(uu, {"label": uu, "role": r2})
                        ek2 = (uu, u, dd.get("relation", ""))
                        if ek2 not in sub_edges_map:
                            sub_edges_map[ek2] = _edge_payload(dd)
                        if r2 == "variant":
                            for uuu, _, ddd in G.in_edges(uu, data=True):
                                if uuu == prot:
                                    sub_nodes.setdefault(prot, {"label": prot, "role": "protein"})
                                    ek3 = (prot, uu, ddd.get("relation", ""))
                                    if ek3 not in sub_edges_map:
                                        sub_edges_map[ek3] = _edge_payload(ddd)
                elif role == "variant":
                    for uu, _, dd in G.in_edges(u, data=True):
                        if uu == prot:
                            sub_nodes.setdefault(prot, {"label": prot, "role": "protein"})
                            ek2 = (prot, u, dd.get("relation", ""))
                            if ek2 not in sub_edges_map:
                                sub_edges_map[ek2] = _edge_payload(dd)

            for _, v, d in G.out_edges(m, data=True):
                if v in members_set:
                    continue
                outgoing[v].add(d.get("relation", ""))
                sub_nodes.setdefault(v, {"label": v, "role": _role(v)})
                ek = (m, v, d.get("relation", ""))
                if ek not in sub_edges_map:
                    sub_edges_map[ek] = _edge_payload(d)

            G.remove_node(m)

        sub_edges = []
        for (s, t, _rel), payload in sub_edges_map.items():
            e = {"source": s, "target": t}
            e.update(payload)
            sub_edges.append(e)

        subgraph_data[gid] = {"nodes": sub_nodes, "edges": sub_edges,
                              "parent_name": parent_name}
        endpoints -= set(members)
        for src, rels in incoming.items():
            for rel in rels:
                G.add_edge(src, gid, relation=rel or "grouped",
                           note=f"{parent_name} ({len(members)} terms)")
        for tgt, rels in outgoing.items():
            for rel in rels:
                G.add_edge(gid, tgt, relation=rel or "grouped",
                           note=f"{parent_name} ({len(members)} terms)")

    # ---------- Kind-aware layered layout with pseudo nodes ----------
    # Layer 0: protein + variants (fixed)
    # Layer -1: endpoints reachable ONLY directly from variants
    # Layers 1..N: longest-path depth plus extra spacing for repeated kinds
    # (e.g. protein -> protein or endpoint -> endpoint).

    def _node_kind(n: str) -> str:
        if n in variants:
            return "variant"
        if motif_kind.get(n) == "onto_group" or n in endpoints:
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
    for gid, kind in motif_kind.items():
        if kind == "onto_group" and gid in G.nodes():
            if _is_direct_only(gid):
                _direct_only_nodes[gid] = -1
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
        elif motif_kind.get(n) == "onto_group":
            members = motif_members.get(n, [])
            depths = [chain_pos.get(m, 1) for m in members if m in chain_pos]
            _init_depth[n] = max(depths) if depths else 1
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
        sp = 140.0 if li == 0 else 170.0
        start = -((n_nodes - 1) * sp) / 2.0
        for i, n in enumerate(ordered):
            x_pos[n] = start + i * sp

    for n in G.nodes():
        if n not in x_pos:
            x_pos[n] = 0.0

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
    rel_color_safe.update({
        "grouped": "#e65100",
    })
    rel_display = {_css_safe(r): r for r in raw_rel_types}

    els = []
    for n, (x, y) in pos.items():
        layer = get_layer(n)
        kind = motif_kind.get(n)

        if n == prot:
            size = 90
        elif kind == "onto_group":
            member_count = len(motif_members.get(n, []))
            size = 60 + min(30, 4 * member_count)
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
        elif kind == "onto_group":
            member_count = len(motif_members.get(n, []))
            parent_name = n.replace("onto_group::", "")
            label = _short_label(f"{parent_name} ({member_count})")
            role_class = "role-endpoint"
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
                "motif_type": kind or "",
                "motif_members": "; ".join(motif_members.get(n, [])),
                "domain_notes": "; ".join(sorted(variant_meta[n]["domain_notes"])) if n in variant_meta else ""
            },
            "classes": f"L{layer} {role_class}" + (f" motif-{kind}" if kind else ""),
            "style": {"width": size, "height": size}
        }
        if n == prot:
            node_el["data"]["uniprot_id"] = protein_uniprot_id or prot
        if n in variants and n in variant_meta:
            vm = variant_meta[n]
            node_el["data"]["gene_symbol"] = prot
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

    legend_rels = ['Gene to Variant'] + raw_rel_types
    legend_colors = {
        'Gene to Variant': '#d5cbc9',
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
    _sp = 170.0
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
    _min_sep = 60.0

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

    legend_rels = ["Gene to Variant"] + raw_rel_types
    legend_colors = {
        "Gene to Variant": "#d5cbc9",
        **{rel_display.get(k, k): v for k, v in rel_color_safe.items()},
    }
    return els, legend_rels, legend_colors, rel_color_safe, list(edge_set), {}


# ------------------------Dash App------------------------–
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.FLATLY])
# Set the server for deployment, see https://dash.plotly.com/deployment
server = app.server
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page")])


def _browse_panel(title: str, helper_text: str, dropdown_id: str, options: list,
                  placeholder: str, button_id: str, directory_id: str,
                  summary_text: str, note_text: Optional[str] = None):
    return html.Div([
        html.H4(title, style={'marginTop': 0, 'marginBottom': 10,
                              'color': '#2c3e50'}),
        html.P(helper_text, style={'fontSize': 16, 'margin': '0 0 16px 0',
                                   'color': '#495057'}),
        dcc.Dropdown(id=dropdown_id, options=options,
                     placeholder=placeholder,
                     style={'fontSize': 18}, clearable=True,
                     searchable=True),
        dbc.Button("Search", id=button_id, n_clicks=0,
                   color="primary", style={'marginTop': 18,
                                           'fontSize': 18}),
        html.Div(summary_text,
                 style={'marginTop': 16, 'marginBottom': 10,
                        'fontSize': 14, 'color': '#6c757d'}),
        *([] if not note_text else [
            html.Div(note_text,
                     style={'marginBottom': 14, 'fontSize': 14,
                            'color': '#6c757d', 'fontStyle': 'italic'})
        ]),
        html.Div(id=directory_id,
                 style={'fontFamily': 'Arial, sans-serif'})
    ], style={'padding': '12px 6px 6px'})


def _render_network_page(view_key: str, root_node_id: str, title: str,
                         graph_tuple: tuple, layout: Optional[dict] = None):
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
            html.Div("Node & Edge Information",
                    style={'fontSize': 18, 'fontWeight': 'bold',
                           'marginBottom': 15, 'color': '#2c3e50',
                           'borderBottom': '2px solid #ecf0f1',
                           'paddingBottom': 10}),
            html.Div("Click on a node or edge to see detailed information, full names, and external links.",
                    style={'color': '#7f8c8d', 'fontSize': 14, 'lineHeight': '1.4'})
        ],
        style={
            'position': 'fixed',
            'left': 0,
            'top': 0,
            'width': 350,
            'height': '100vh',
            'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
            'padding': 20,
            'boxShadow': '2px 0 10px rgba(0,0,0,0.1)',
            'borderRight': '1px solid #dee2e6',
            'fontSize': 16,
            'fontFamily': 'Arial, sans-serif',
            'zIndex': 1000,
            'overflowY': 'auto'
        }
    )

    main_content = html.Div([
        html.Div([
            dcc.Link("← Home", href="/",
                    style={'color': '#0366d6', 'textDecoration': 'none',
                           'fontSize': 15, 'fontWeight': 'bold'}),
            html.H4(title,
                   style={'textAlign': 'center', 'margin': '2px 0',
                          'color': '#2c3e50'}),
            html.P("Tip: click the root node for this view to clear all highlights.",
                   style={'textAlign': 'center', 'marginTop': 0,
                          'marginBottom': 15, 'color': '#666',
                          'fontFamily': 'Arial, sans-serif', 'fontSize': 14})
        ], style={'padding': '10px 10px', 'background': '#ffffff',
                  'borderBottom': '1px solid #dee2e6'}),

        dcc.Store(id={'type': 'store-els',  'prot': view_key},  data=els),
        dcc.Store(id={'type': 'store-edges', 'prot': view_key},  data=edge_set),
        dcc.Store(id={'type': 'store-root', 'prot': view_key},  data=root_node_id),
        dcc.Store(id={'type': 'store-subgraphs', 'prot': view_key}, data=subgraph_data),
        dcc.Store(id={'type': 'store-relcolors', 'prot': view_key}, data=rel_color_safe),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                id={'type': 'subgraph-title', 'prot': view_key})),
            dbc.ModalBody(
                html.Div([
                    cyto.Cytoscape(
                        id={'type': 'cy-subgraph', 'prot': view_key},
                        elements=[], layout={'name': 'preset'},
                        style={'width': '70%', 'height': '100%'},
                        stylesheet=[]),
                    html.Div(
                        id={'type': 'subgraph-edge-info', 'prot': view_key},
                        children=[
                            html.Div("Click an edge to see details",
                                     style={'color': '#7f8c8d', 'fontSize': 14,
                                            'padding': 16})
                        ],
                        style={'width': '30%', 'height': '100%',
                               'overflowY': 'auto', 'borderLeft': '1px solid #dee2e6',
                               'background': '#f8f9fa'})
                ], style={'display': 'flex', 'height': '70vh'}),
                style={'padding': 0}),
        ], id={'type': 'subgraph-modal', 'prot': view_key},
           size="xl", is_open=False),

        cyto.Cytoscape(
            id={'type': 'cy-net', 'prot': view_key},
            elements=els,
            layout=layout,
            style={'width': '100%', 'height': 'calc(100vh - 120px)'},
            stylesheet=[
                {'selector': 'node', 'style': {
                    'shape': 'ellipse', 'background-opacity': 0.5,
                    'font-size': 16, 'font-weight': 'bold',
                    'label': 'data(label)',
                    'text-wrap': 'wrap',
                    'text-max-width': 100,
                    'text-valign': 'center',
                    'text-halign': 'center'}},
                {'selector': '.role-protein',
                 'style': {'background-color': '#aacdd7', 'color': '#004466'}},
                {'selector': '.role-variant',
                 'style': {'background-color': '#a492bb', 'color': '#573d82'}},
                {'selector': '.role-intermediate',
                 'style': {'background-color': '#cce9b6', 'color': '#3f6330'}},
                {'selector': '.role-endpoint',
                 'style': {'background-color': '#fabf77', 'color': '#b05e04'}},
                {'selector': '.motif-onto_group',
                 'style': {'shape': 'round-rectangle',
                           'background-color': '#ffcc80',
                           'background-opacity': 0.7,
                           'border-width': 2,
                           'border-color': '#e65100'}},
                {'selector': '.edge-PV',
                 'style': {'line-color': '#d5cbc9',
                           'target-arrow-shape': 'triangle',
                           'width': 2}},
                *[rel_style(css_cls, c) for css_cls, c in rel_color_safe.items()],
                {'selector': '.faded', 'style': {'opacity': 0.15}}
            ]),

        html.Div([
            html.H4("Legend",
                    style={'margin': 0, 'fontSize': 16,
                           'fontWeight': 'bold',
                           'fontFamily': 'Arial, sans-serif',
                           'color': '#2c3e50'}),
            html.Ul([
                html.Li([html.Span('→',
                                  style={'color': legend_colors.get(r, '#d5cbc9'),
                                         'marginRight': 8,
                                         'fontSize': 16}), r],
                        style={'fontSize': 14, 'listStyle': 'none',
                               'margin': '2px 0'})
                for r in legend_rels
            ], style={'paddingLeft': 0, 'margin': '8px 0 0 0'})
        ], style={'position': 'absolute', 'top': 20, 'right': 20,
                  'background': 'rgba(255,255,255,0.95)',
                  'padding': '12px 16px',
                  'borderRadius': 8,
                  'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                  'fontFamily': 'Arial, sans-serif',
                  'maxHeight': '70vh',
                  'overflowY': 'auto'})

    ], style={
        'marginLeft': 350,
        'position': 'relative',
        'height': '100vh'
    })

    return html.Div([sidebar, main_content])


# ---Homepage---
def homepage():
    search_card = html.Div(
        [
            html.H1("Variant Network Explorer",
                    style={'marginTop': 0, 'marginBottom': 12}),
            html.P("Browse either gene-centric or endpoint-centric variant networks.",
                   style={'fontSize': 18, 'margin': '0 0 20px 0'}),
            dcc.Tabs([
                dcc.Tab(
                    label="Gene-centric",
                    children=[
                        _browse_panel(
                            title="Protein / Gene",
                            helper_text="Type a protein or gene name below, or browse alphabetically.",
                            dropdown_id='prot-search',
                            options=PROT_OPTIONS,
                            placeholder="search protein / gene …",
                            button_id='submit-prot',
                            directory_id='prot-directory',
                            summary_text=f"{len(PROTS)} protein/gene pages available.",
                        )
                    ]
                ),
                dcc.Tab(
                    label="Endpoint-centric",
                    children=[
                        _browse_panel(
                            title="Disease / Result / Endpoint",
                            helper_text="Type an end node from the current graph, or browse alphabetically.",
                            dropdown_id='endpoint-search',
                            options=ENDPOINT_OPTIONS,
                            placeholder="search disease / result / endpoint …",
                            button_id='submit-endpoint',
                            directory_id='endpoint-directory',
                            summary_text=(
                                f"{len(ENDPOINTS)} unique end nodes currently indexed "
                                f"from biological_process/disease."
                            ),
                            note_text=(
                                "Large endpoint pages are aggregated at the protein level "
                                "so common diseases/results stay explorable."
                            ),
                        )
                    ]
                )
            ])
        ],
        style={'maxWidth': 880, 'margin': '40px auto',
               'background': '#f8f9fa', 'padding': '32px 48px',
               'borderRadius': 8, 'boxShadow': '0 0 8px rgba(0,0,0,0.15)',
               'fontFamily': 'Arial, sans-serif'}
    )

    footer = html.Div(
        [
            html.Span("Developed by the "),
            html.A("Gyori Lab", href="https://gyorilab.github.io",
                   target="_blank"),
            html.Span(" at Northeastern University"),
            html.Br(),
            html.Span("INDRA Variant is funded under DARPA ASKEM / "
                      "ARPA-H BDF (HR00112220036)")
        ],
        style={'background': '#f1f1f1', 'padding': '10px 24px',
               'textAlign': 'center', 'fontSize': 14,
               'fontFamily': 'Arial, sans-serif', 'marginTop': 40}
    )

    return html.Div([search_card, footer])


# ---Network Page---
def network_page(prot: str):
    return _render_network_page(
        view_key=f"protein::{prot}",
        root_node_id=prot,
        title=f"{prot} Variant Network",
        graph_tuple=build_elements(prot),
    )


def endpoint_network_page(endpoint: str):
    return _render_network_page(
        view_key=f"endpoint::{endpoint}",
        root_node_id=endpoint,
        title=f"{endpoint} Endpoint-Centric Network",
        graph_tuple=build_endpoint_elements(endpoint),
        layout={'name': 'preset'},
    )


# ----
@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(path):
    if path in (None, "/"):
        return homepage()
    if path.startswith("/protein/"):
        prot = _decode_route_value(path.split("/protein/", 1)[1])
        if prot in PROTS:
            return network_page(prot)
    if path.startswith("/endpoint/"):
        endpoint = _decode_route_value(path.split("/endpoint/", 1)[1])
        if endpoint in ENDPOINT_INDEX:
            return endpoint_network_page(endpoint)
    return html.H3("404 – Not found")

def _sidebar_default():
    return [
        html.Div("Node & Edge Information",
                 style={'fontSize': 18, 'fontWeight': 'bold',
                        'marginBottom': 15, 'color': '#2c3e50',
                        'borderBottom': '2px solid #ecf0f1',
                        'paddingBottom': 10}),
        html.Div("Click on a node or edge to see detailed information, full names, and external links.",
                 style={'color': '#7f8c8d', 'fontSize': 14, 'lineHeight': '1.4'})
    ]


def _sidebar_card(children):
    return html.Div(children, style={
        'background': '#ffffff',
        'padding': 12,
        'borderRadius': 6,
        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'marginBottom': 15,
        'border': '1px solid #dee2e6'
    })


def _build_edge_info(edge):
    if not edge:
        return _sidebar_default()

    content = [
        html.Div("Edge Information",
                 style={'fontSize': 18, 'fontWeight': 'bold',
                        'marginBottom': 15, 'color': '#2c3e50',
                        'borderBottom': '2px solid #ecf0f1',
                        'paddingBottom': 10})
    ]

    rel = edge.get('rel', 'N/A')
    source = edge.get('source', 'N/A')
    target = edge.get('target', 'N/A')

    content.append(_sidebar_card([
        html.Div("Relationship",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': '#34495e', 'marginBottom': 5}),
        html.Div(f"{source} → {target}",
                 style={'fontSize': 14, 'color': '#2c3e50',
                        'marginBottom': 8, 'fontWeight': 'bold'}),
        *([] if rel in ['DV', 'PV', 'has_domain'] else [
            html.Div(f"Type: {rel}",
                     style={'fontSize': 14, 'color': '#7f8c8d'})
        ]),
        *([] if not edge.get('evidence_count') else [
            html.Div(f"Evidence count: {edge['evidence_count']}",
                     style={'fontSize': 13, 'color': '#7f8c8d', 'marginTop': 4})
        ])
    ]))

    if rel == 'DV':
        if edge.get('note'):
            content.append(_sidebar_card([
                html.Div("Domain Description",
                         style={'fontSize': 14, 'fontWeight': 'bold',
                                'color': '#34495e', 'marginBottom': 8}),
                html.Div(edge['note'],
                         style={'fontSize': 14, 'color': '#2c3e50',
                                'lineHeight': '1.4'})
            ]))
    else:
        if edge.get('note'):
            content.append(_sidebar_card([
                html.Div("Description",
                         style={'fontSize': 14, 'fontWeight': 'bold',
                                'color': '#34495e', 'marginBottom': 8}),
                html.Div(edge['note'],
                         style={'fontSize': 14, 'color': '#2c3e50',
                                'lineHeight': '1.4'})
            ]))

    if edge.get('clinvar_data'):
        data = edge['clinvar_data']
        content.append(_sidebar_card([
            html.Div("ClinVar Information",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 10}),
            html.Div([
                html.Div([
                    html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('pathogenicity', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Review Status: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('review', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Associated Condition: ", style={'fontWeight': 'bold'}),
                    html.Div(data.get('conditions', 'N/A'),
                             style={'marginTop': 4, 'fontStyle': 'italic'})
                ])
            ], style={'fontSize': 13, 'color': '#2c3e50', 'lineHeight': '1.4'})
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
                              'marginBottom': 4, 'color': '#0366d6',
                              'textDecoration': 'none', 'fontSize': 13})
            )
        content.append(_sidebar_card([
            html.Div("External Resources",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 10}),
            html.Div(pubmed_links,
                     style={'marginBottom': 8, 'lineHeight': '1.8'}),
            html.A("View in INDRA",
                   href=(f"https://discovery.indra.bio/search/"
                         f"?agent={_url.quote_plus(edge['src4indra'])}"
                         f"&other_agent={_url.quote_plus(edge['target'])}"
                         "&agent_role=subject&other_role=object"),
                   target="_blank",
                   style={'display': 'block', 'color': '#0366d6',
                          'textDecoration': 'none', 'fontSize': 13,
                          'fontWeight': 'bold'})
        ]))

    return content


def _build_node_info(node):
    if not node:
        return _sidebar_default()

    real_name = node.get('real', node.get('label', 'N/A'))
    role = node.get('role', 'unknown').replace('-', ' ').replace('_', ' ').title()
    if node.get('motif_type') == 'onto_group':
        role = "Endpoint Group"

    content = [
        html.Div("Node Information",
                 style={'fontSize': 18, 'fontWeight': 'bold',
                        'marginBottom': 15, 'color': '#2c3e50',
                        'borderBottom': '2px solid #ecf0f1',
                        'paddingBottom': 10})
    ]

    content.append(_sidebar_card([
        html.Div("Full Name",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': '#34495e', 'marginBottom': 5}),
        html.Div(real_name,
                 style={'fontSize': 14, 'color': '#2c3e50',
                        'marginBottom': 8, 'fontWeight': 'bold',
                        'lineHeight': '1.4'}),
        html.Div(f"Role: {role}",
                 style={'fontSize': 13, 'color': '#7f8c8d'})
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
            html.Span("Supporting records: ", style={'fontWeight': 'bold'}),
            html.Span(str(node['n_records']))
        ]))
    if stat_lines:
        content.append(_sidebar_card([
            html.Div("Summary",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 10}),
            *stat_lines
        ]))

    if node.get('domain_notes'):
        content.append(_sidebar_card([
            html.Div("Domain Notes",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 8}),
            html.Div(node['domain_notes'],
                     style={'fontSize': 14, 'color': '#2c3e50',
                            'lineHeight': '1.4'})
        ]))

    if node.get('clinvar_data'):
        data = node['clinvar_data']
        content.append(_sidebar_card([
            html.Div("ClinVar Information",
                     style={'fontSize': 14, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 10}),
            html.Div([
                html.Div([
                    html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('pathogenicity', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Review Status: ", style={'fontWeight': 'bold'}),
                    html.Span(data.get('review', 'N/A'))
                ], style={'marginBottom': 6}),
                html.Div([
                    html.Span("Associated Condition: ", style={'fontWeight': 'bold'}),
                    html.Div(data.get('conditions', 'N/A'),
                             style={'marginTop': 4, 'fontStyle': 'italic'})
                ])
            ], style={'fontSize': 13, 'color': '#2c3e50', 'lineHeight': '1.4'})
        ]))

    external_links = [
        html.A("Search in INDRA",
               href=f"https://discovery.indra.bio/search/?agent={_url.quote_plus(real_name)}",
               target="_blank",
               style={'display': 'block', 'color': '#0366d6',
                      'textDecoration': 'none', 'fontSize': 13,
                      'fontWeight': 'bold', 'marginBottom': 6})
    ]
    if node.get('protein_page'):
        external_links.insert(
            0,
            dcc.Link("Open protein-centric view",
                     href=node['protein_page'],
                     style={'display': 'block', 'color': '#0366d6',
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
            html.A("View in UniProt",
                   href=uniprot_href,
                   target="_blank",
                   style={'display': 'block', 'color': '#0366d6',
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    if node.get('clinvar_allele'):
        external_links.append(
            html.A("View in ClinVar",
                   href=("https://www.ncbi.nlm.nih.gov/clinvar/?term="
                         f"{_url.quote_plus(str(node['clinvar_allele']) + '[alleleid]')}"),
                   target="_blank",
                   style={'display': 'block', 'color': '#0366d6',
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    elif node.get('gene_symbol'):
        clinvar_query = f"{node['gene_symbol']}[gene] AND {real_name}"
        external_links.append(
            html.A("Search ClinVar",
                   href=("https://www.ncbi.nlm.nih.gov/clinvar/?term="
                         f"{_url.quote_plus(clinvar_query)}"),
                   target="_blank",
                   style={'display': 'block', 'color': '#0366d6',
                          'textDecoration': 'none', 'fontSize': 13,
                          'marginBottom': 6})
        )
    if node.get('dbsnp_rs'):
        external_links.append(
            html.A("View in dbSNP",
                   href=f"https://www.ncbi.nlm.nih.gov/snp/rs{node['dbsnp_rs']}",
                   target="_blank",
                   style={'display': 'block', 'color': '#0366d6',
                          'textDecoration': 'none', 'fontSize': 13})
        )

    content.append(_sidebar_card([
        html.Div("External Resources",
                 style={'fontSize': 14, 'fontWeight': 'bold',
                        'color': '#34495e', 'marginBottom': 10}),
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
            "background-opacity": 0.7,
            "text-wrap": "wrap", "text-max-width": 110}},
        {"selector": ".role-protein", "style": {
            "background-color": "#aacdd7", "color": "#004466"}},
        {"selector": ".role-variant", "style": {
            "background-color": "#a492bb", "color": "#573d82"}},
        {"selector": ".role-intermediate", "style": {
            "background-color": "#cce9b6", "color": "#3f6330"}},
        {"selector": ".role-endpoint", "style": {
            "background-color": "#fabf77", "color": "#b05e04"}},
        {"selector": ".edge-PV", "style": {
            "line-color": "#d5cbc9", "target-arrow-color": "#d5cbc9",
            "target-arrow-shape": "triangle", "curve-style": "bezier",
            "width": 1.5, "label": "data(label)",
            "font-size": 9, "color": "#d5cbc9", "text-rotation": "autorotate"}},
    ]
    for css_cls, color in rel_colors.items():
        stylesheet.append({
            "selector": f".edge-{css_cls}", "style": {
                "line-color": color, "target-arrow-color": color,
                "target-arrow-shape": "triangle", "curve-style": "bezier",
                "width": 2, "label": "data(label)",
                "font-size": 9, "color": color, "text-rotation": "autorotate"}
        })

    title = f"{parent_name}  ({n_endpoints} members)"
    return True, title, sub_els, stylesheet

# ---------------------- subgraph edge-info callback ------------------------
@app.callback(
    Output({'type': 'subgraph-edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-subgraph', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_subgraph_edge_info(edge):
    if not edge:
        return [html.Div("Click an edge to see details",
                         style={'color': '#7f8c8d', 'fontSize': 14, 'padding': 16})]

    rel = edge.get('rel', '')
    source = edge.get('source', '')
    target = edge.get('target', '')

    card_style = {'background': '#ffffff', 'padding': 12, 'borderRadius': 6,
                  'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                  'marginBottom': 12, 'border': '1px solid #dee2e6'}

    content = [
        html.Div("Edge Detail",
                 style={'fontSize': 16, 'fontWeight': 'bold', 'color': '#2c3e50',
                        'borderBottom': '2px solid #ecf0f1', 'paddingBottom': 8,
                        'marginBottom': 12, 'padding': '12px 16px 8px'}),
        html.Div([
            html.Div(f"{source} → {target}",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': '#2c3e50', 'marginBottom': 6}),
            *([] if rel in ('PV',) else [
                html.Div(f"Type: {rel}",
                         style={'fontSize': 13, 'color': '#7f8c8d'})]),
            *([] if not edge.get('evidence_count') else [
                html.Div(f"Evidence count: {edge['evidence_count']}",
                         style={'fontSize': 12, 'color': '#7f8c8d', 'marginTop': 4})]),
        ], style={**card_style, 'margin': '0 12px 12px'}),
    ]

    if edge.get('note'):
        content.append(html.Div([
            html.Div("Description",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 6}),
            html.Div(edge['note'],
                     style={'fontSize': 13, 'color': '#2c3e50', 'lineHeight': '1.4'})
        ], style={**card_style, 'margin': '0 12px 12px'}))

    if edge.get('clinvar_data'):
        data = edge['clinvar_data']
        content.append(html.Div([
            html.Div("ClinVar",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 6}),
            html.Div([
                html.Span("Pathogenicity: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('pathogenicity', 'N/A'))
            ], style={'fontSize': 12, 'marginBottom': 4}),
            html.Div([
                html.Span("Review: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('review', ''))
            ], style={'fontSize': 12, 'marginBottom': 4}),
            html.Div([
                html.Span("Conditions: ", style={'fontWeight': 'bold'}),
                html.Span(data.get('conditions', ''))
            ], style={'fontSize': 12, 'lineHeight': '1.4'}),
        ], style={**card_style, 'margin': '0 12px 12px'}))

    if edge.get('pmid'):
        raw = str(edge['pmid'])
        pmids = [p.strip() for p in re.split(r'[;,]', raw) if p.strip()]
        links = []
        for p in pmids:
            links.append(html.A(
                f"PMID:{p}",
                href=f"https://pubmed.ncbi.nlm.nih.gov/{p}/",
                target="_blank",
                style={'display': 'block', 'color': '#0366d6',
                       'textDecoration': 'none', 'fontSize': 12,
                       'marginBottom': 3}))
        content.append(html.Div([
            html.Div("PubMed",
                     style={'fontSize': 13, 'fontWeight': 'bold',
                            'color': '#34495e', 'marginBottom': 6}),
            *links
        ], style={**card_style, 'margin': '0 12px 12px'}))

    return content

# ---------------------- highlight callback ---------------------------------
@app.callback(
    Output({'type': 'cy-net', 'prot': MATCH}, 'elements'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapNodeData'),
    [State({'type': 'store-els',   'prot': MATCH}, 'data'),
     State({'type': 'store-edges', 'prot': MATCH}, 'data'),
     State({'type': 'store-root',  'prot': MATCH}, 'data')],
    prevent_initial_call=True)
def highlight(node, elements, edge_set, root_prot):
    if not node:
        return elements

    if node['id'] == root_prot:
        for el in elements:
            el['classes'] = el['classes'].replace(' faded', '')
        return elements

    edge_set = {tuple(e) for e in edge_set}
    sel = node['id']
    keep_nodes = {sel}
    keep_edges = set()

    stack = [sel]
    while stack:
        cur = stack.pop()
        for s, t in edge_set:
            if s == cur and (s, t) not in keep_edges:
                keep_edges.add((s, t))
                keep_nodes.add(t)
                stack.append(t)

    stack = [sel]
    while stack:
        cur = stack.pop()
        for s, t in edge_set:
            if t == cur and (s, t) not in keep_edges:
                keep_edges.add((s, t))
                keep_nodes.add(s)
                stack.append(s)

    for el in elements:
        if 'source' in el['data']:  # edge
            keep = ((el['data']['source'], el['data']['target']) in keep_edges
                    or el['data']['rel'] == 'PV')
        else:  # node
            keep = el['data']['id'] in keep_nodes

        if keep:
            el['classes'] = el['classes'].replace(' faded', '')
        else:
            if 'faded' not in el['classes']:
                el['classes'] += ' faded'

    return elements


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


# --------------------Run App----------------------------–
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
