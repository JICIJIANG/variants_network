import re
import math
import itertools as _it
import logging
import time as _time
import urllib.parse as _url
from pathlib import Path
from collections import defaultdict
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
PROTS = sorted(TSV_RE.match(p.name).group("prot")
               for p in Path(DATA_DIR).iterdir()
               if TSV_RE.match(p.name))
AA_SUB_RE = re.compile(r"^[A-Za-z\*]+(?P<pos>\d+)[A-Za-z\*=]+$")
P_DOT_RE = re.compile(r"p\.[A-Za-z\*]+(?P<pos>\d+)[A-Za-z\*=]+", re.I)

# ---------- OBO ontology cache (pre-built JSON) ----------
import json as _json

_OBO_CACHE_PATH = Path(DATA_DIR).parent.parent / "obo_endpoint_cache.json"

if _OBO_CACHE_PATH.exists():
    with open(_OBO_CACHE_PATH) as _f:
        _OBO_CACHE = _json.load(_f)
else:
    _OBO_CACHE = {}


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
    df_path = Path(DATA_DIR) / f"{prot}_variant_effects_with_clinvar_with_domains.tsv"
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

    # ---------- Longest-path layered layout with pseudo nodes ----------
    # Layer 0: protein + variants (fixed)
    # Layer -1: endpoints reachable ONLY directly from variants
    # Layers 1..N: determined by longest path distance from any variant

    # -- Step 1: layer assignment from chain positions ---------------------
    node_depth: dict[str, int] = {}
    for v in variants:
        node_depth[v] = 0
    node_depth[prot] = 0

    _init_depth: dict[str, int] = {}
    for n in G.nodes():
        if n == prot or n in variants:
            continue
        if n in chain_pos:
            _init_depth[n] = chain_pos[n]
        elif motif_kind.get(n) == "onto_group":
            members = motif_members.get(n, [])
            depths = [chain_pos.get(m, 1) for m in members if m in chain_pos]
            _init_depth[n] = max(depths) if depths else 1
        else:
            _init_depth[n] = 1

    # -- Step 2: forward propagation on condensation DAG (cycle-safe) ---
    #    Collapse SCCs so cycles don't cascade layer depths.
    _non_vp = [n for n in G.nodes() if n != prot and n not in variants]
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
        for sid in nx.topological_sort(_C):
            for succ_sid in _C.successors(sid):
                if _scc_depth.get(succ_sid, 0) <= _scc_depth.get(sid, 0):
                    _scc_depth[succ_sid] = _scc_depth[sid] + 1
        for n in _non_vp:
            node_depth[n] = _scc_depth[_scc_map[n]]
    else:
        for n, d in _init_depth.items():
            node_depth[n] = d

    # -- Step 3: identify "direct-only" endpoints → layer -1 -----------
    _direct_only_eps = set()
    for ep in endpoints:
        preds = {u for u, _, _ in G.in_edges(ep, data=True) if u != prot}
        if preds and all(p in variants for p in preds):
            _direct_only_eps.add(ep)
    for ep in _direct_only_eps & set(G.nodes()):
        node_depth[ep] = -1

    for gid, kind in motif_kind.items():
        if kind == "onto_group" and gid in G.nodes():
            preds = {u for u, _, _ in G.in_edges(gid, data=True) if u != prot}
            if preds and all(p in variants for p in preds):
                node_depth[gid] = -1

    pseudo_nodes = set()

    def get_layer(n):
        if n == prot or n in variants:
            return 0
        return node_depth.get(n, 1)

    # -- Step 4: collect all layers ------------------------------------
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

    # -- Step 6: assign positions from optimised orderings -------------
    x_pos: dict[str, float] = {}
    sorted_layer_keys = sorted(optimized.keys())
    max_layer = max(sorted_layer_keys) if sorted_layer_keys else 0
    min_layer = min(sorted_layer_keys) if sorted_layer_keys else 0
    layer_gap = 190.0
    y_pos: dict[int, float] = {}
    for li in sorted_layer_keys:
        y_pos[li] = li * layer_gap

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
        edge_data={
            "id" :f"{u}->{v}_{d.get('pmid', '')}_{d.get('note', '')}",
            "source": u,
            "target": v,
            "rel": relation,
            "src4indra": src4indra
        }
        if 'pmid' in d and d['pmid']:
            edge_data['pmid'] = d['pmid']
        if 'note' in d and d['note']:
            edge_data['note'] = d['note']
        if 'clinvar_data' in d and d['clinvar_data']:
            edge_data['clinvar_data'] = d['clinvar_data']
        if 'weight' in d:
            edge_data['evidence_count'] = d['weight']

        els.append({
            "data": edge_data,
            "classes": cls
        })

    edge_set = {(u, v) for u, v in G.edges()}
    legend_rels = ['Gene to Variant'] + raw_rel_types
    legend_colors = {
        'Gene to Variant': '#d5cbc9',
        **{rel_display.get(k, k): v for k, v in rel_color_safe.items()}
    }

    return els, legend_rels, legend_colors, rel_color_safe, list(edge_set), subgraph_data


# ------------------------Dash App------------------------–
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.FLATLY])
# Set the server for deployment, see https://dash.plotly.com/deployment
server = app.server
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page")])


# ---Homepage---
def homepage():
    prot_options = [{'label': p, 'value': p} for p in PROTS]

    search_card = html.Div(
        [
            html.H1("Protein Variant Network Explorer",
                    style={'marginTop': 0, 'marginBottom': 12}),
            html.P("Type a protein/gene name below (auto-suggest enabled) "
                   "or browse alphabetically.",
                   style={'fontSize': 18, 'margin': '0 0 20px 0'}),
            dcc.Dropdown(id='prot-search', options=prot_options,
                         placeholder="search protein / gene …",
                         style={'fontSize': 18}, clearable=True,
                         searchable=True),
            dbc.Button("Search", id='submit-prot', n_clicks=0,
                       color="primary", style={'marginTop': 18,
                                               'fontSize': 18}),
        ],
        style={'maxWidth': 880, 'margin': '40px auto',
               'background': '#f8f9fa', 'padding': '32px 48px',
               'borderRadius': 8, 'boxShadow': '0 0 8px rgba(0,0,0,0.15)',
               'fontFamily': 'Arial, sans-serif'}
    )

    directory = html.Div(id='prot-directory',
                         style={'maxWidth': 880, 'margin': '0 auto',
                                'fontFamily': 'Arial, sans-serif'})

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

    return html.Div([search_card, directory, footer])


# ---Network Page---
def network_page(prot: str):
    els, legend_rels, legend_colors, rel_color_safe, edge_set, subgraph_data = build_elements(prot)

    def rel_style(css_cls, c):
        return {'selector': f'.edge-{css_cls}',
                'style': {'line-color': c, 'target-arrow-color': c,
                          'target-arrow-shape': 'triangle',
                          'curve-style': 'bezier', 'width': 2}}

    sidebar = html.Div(
        id={'type': 'edge-info', 'prot': prot},
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
            html.H4(f"{prot} Variant Network", 
                   style={'textAlign': 'center', 'margin': '2px 0',
                          'color': '#2c3e50'}),
            html.P("Tip: click the central protein/gene to clear all highlights.",
                   style={'textAlign': 'center', 'marginTop': 0,
                          'marginBottom': 15, 'color': '#666',
                          'fontFamily': 'Arial, sans-serif', 'fontSize': 14})
        ], style={'padding': '10px 10px', 'background': '#ffffff',
                  'borderBottom': '1px solid #dee2e6'}),

        dcc.Store(id={'type': 'store-els',  'prot': prot},  data=els),
        dcc.Store(id={'type': 'store-edges', 'prot': prot},  data=edge_set),
        dcc.Store(id={'type': 'store-root', 'prot': prot},  data=prot),
        dcc.Store(id={'type': 'store-subgraphs', 'prot': prot}, data=subgraph_data),
        dcc.Store(id={'type': 'store-relcolors', 'prot': prot}, data=rel_color_safe),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(
                id={'type': 'subgraph-title', 'prot': prot})),
            dbc.ModalBody(
                html.Div([
                    cyto.Cytoscape(
                        id={'type': 'cy-subgraph', 'prot': prot},
                        elements=[], layout={'name': 'preset'},
                        style={'width': '70%', 'height': '100%'},
                        stylesheet=[]),
                    html.Div(
                        id={'type': 'subgraph-edge-info', 'prot': prot},
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
        ], id={'type': 'subgraph-modal', 'prot': prot},
           size="xl", is_open=False),

        cyto.Cytoscape(
            id={'type': 'cy-net', 'prot': prot},
            elements=els, 
            layout={'name': 'preset'},
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
                # Role classes for readability.
                {'selector': '.role-protein',
                 'style': {'background-color': '#aacdd7', 'color': '#004466'}},
                {'selector': '.role-variant',
                 'style': {'background-color': '#a492bb', 'color': '#573d82'}},
                {'selector': '.role-intermediate',
                 'style': {'background-color': '#cce9b6', 'color': '#3f6330'}},
                {'selector': '.role-endpoint',
                 'style': {'background-color': '#fabf77', 'color': '#b05e04'}},
                {'selector': '.role-pseudo',
                 'style': {'background-color': '#cccccc',
                           'background-opacity': 0.6,
                           'shape': 'ellipse',
                           'width': 12, 'height': 12,
                           'label': ''}},
                # Ontology group nodes
                {'selector': '.motif-onto_group',
                 'style': {'shape': 'round-rectangle',
                           'background-color': '#ffcc80',
                           'background-opacity': 0.7,
                           'border-width': 2,
                           'border-color': '#e65100'}},
                # Edges
                {'selector': '.edge-PV', 
                 'style': {'line-color': '#d5cbc9', 
                           'target-arrow-shape': 'triangle', 
                           'width': 2}},
                
                *[rel_style(css_cls, c) for css_cls, c in rel_color_safe.items()],
                {'selector': '.edge-pseudo', 'style': {
                    'line-style': 'dotted', 'width': 1.5,
                    'target-arrow-shape': 'none'}},
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


# ----
@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(path):
    if path in (None, "/"):
        return homepage()
    if path.startswith("/protein/"):
        prot = path.split("/")[2]
        if prot in PROTS:
            return network_page(prot)
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
    query = (query or "").strip().lower()
    blocks = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        group = [p for p in PROTS if p.startswith(letter)]
        if query:
            group = [p for p in group if query in p.lower()]
        if not group:
            continue
        blocks.append(
            html.Details([
                html.Summary(f"{letter} ({len(group)})",
                             style={'cursor': 'pointer', 'fontSize': 20}),
                html.Ul([
                    html.Li(html.A(p, href=f"/protein/{p}",
                                   style={'textDecoration': 'none',
                                          'color': '#0366d6',
                                          'fontWeight': 'bold'
                                          if query and query in p.lower()
                                          else 'normal'}))
                    for p in group
                ], style={'columnCount': 3, 'listStyle': 'none',
                          'padding': 0, 'margin': '6px 0'})
            ], open=bool(query))
        )
    return blocks


@app.callback(Output('url', 'href', allow_duplicate=True),
              Input('submit-prot', 'n_clicks'),
              State('prot-search', 'value'),
              prevent_initial_call=True)
def jump_to_protein(_, value):
    return f"/protein/{value}" if value else dash.no_update


# --------------------Run App----------------------------–
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
