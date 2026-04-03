import re
import math
import urllib.parse as _url
from pathlib import Path
from collections import defaultdict
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
from dash import html, dcc, Input, Output, State, MATCH

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

    variant_meta = defaultdict(lambda: {
        "domains": set(),
        "domain_notes": set(),
        "clinvar_data": None,
        "protein_pos": None
    })
    edge_bucket = defaultdict(lambda: {
        "pmids": set(),
        "notes": set(),
        "clinvar_data": None,
        "count": 0
    })

    for _, row in df.iterrows():
        var = row["variant_info"]
        name_label = row.get("Name", "")
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
        for seg in str(row.get("chain", "")).split(" -[")[1:]:
            if "]->" not in seg:
                continue
            rel, tgt = seg.split("]->", 1)
            rel = rel.strip()
            tgt = tgt.strip()
            if not tgt:
                continue
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

    # ---------- Layered layout with semi-fixed baseline ----------
    def get_layer(n):
        if motif_kind.get(n) == "onto_group":
            return 2
        if n == prot or n in variants:
            return 0
        if n in endpoints:
            return 2
        return 1

    layers = {0: [], 1: [], 2: []}
    for n in G.nodes():
        layers[get_layer(n)].append(n)

    def variant_sort_key(node_id: str):
        if motif_kind.get(node_id) == "fan":
            pos_values = [variant_meta[m]["protein_pos"] for m in motif_members.get(node_id, [])]
            pos_values = [p for p in pos_values if p is not None]
            pos = sum(pos_values) / len(pos_values) if pos_values else float("inf")
            return (0, pos, node_id)
        if node_id in variants:
            pos = variant_meta[node_id]["protein_pos"]
            return (0, pos if pos is not None else float("inf"), node_id)
        return (1, float("inf"), node_id)

    baseline_nodes = [n for n in layers[0] if n != prot]
    baseline_nodes.sort(key=variant_sort_key)
    baseline_spacing = 140.0
    x_pos = {}
    y_pos = {0: 280.0, 1: 90.0, 2: -80.0}
    for i, n in enumerate(baseline_nodes):
        x_pos[n] = i * baseline_spacing
    if baseline_nodes:
        x_pos[prot] = x_pos[baseline_nodes[-1]] + 260.0
    else:
        x_pos[prot] = 0.0

    def adjacent_values(node_id: str, candidate_layer_values: dict):
        vals = []
        for u, _, _ in G.in_edges(node_id, data=True):
            if u in candidate_layer_values:
                vals.append(candidate_layer_values[u])
            elif u in x_pos:
                vals.append(x_pos[u])
        for _, v, _ in G.out_edges(node_id, data=True):
            if v in candidate_layer_values:
                vals.append(candidate_layer_values[v])
            elif v in x_pos:
                vals.append(x_pos[v])
        return vals

    def assign_with_spacing(node_order, target_map, spacing):
        if not node_order:
            return
        center = sum(x_pos[n] for n in baseline_nodes) / len(baseline_nodes) if baseline_nodes else 0.0
        start = center - ((len(node_order) - 1) * spacing / 2.0)
        for idx, n in enumerate(node_order):
            target_map[n] = start + idx * spacing

    # Initialize order for upper layers and refine by barycentric sweeps.
    l1 = sorted(layers[1])
    l2 = sorted(layers[2])
    assign_with_spacing(l1, x_pos, 170.0)
    assign_with_spacing(l2, x_pos, 200.0)

    def barycenter_key(n):
        vals = adjacent_values(n, x_pos)
        return (sum(vals) / max(len(vals), 1), n)

    for _ in range(8):
        l1.sort(key=barycenter_key)
        assign_with_spacing(l1, x_pos, 170.0)
        l2.sort(key=barycenter_key)
        assign_with_spacing(l2, x_pos, 200.0)

    pos = {n: (x_pos.get(n, 0.0), y_pos[get_layer(n)]) for n in G.nodes()}

    def _css_safe(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9_-]', '-', name)

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
        elif layer == 2:
            size = 52 + min(24, 5 * endpoint_freq.get(n, 1))
        else:
            size = 54

        label = n
        if kind == "onto_group":
            member_count = len(motif_members.get(n, []))
            parent_name = n.replace("onto_group::", "")
            label = f"{parent_name} ({member_count})"

        if n == prot:
            role_class = "role-protein"
        elif n in variants:
            role_class = "role-variant"
        elif kind == "onto_group":
            role_class = "role-endpoint"
        elif layer == 2:
            role_class = "role-endpoint"
        else:
            role_class = "role-intermediate"

        node_el = {
            "data": {
                "id": n,
                "label": label,
                "real": n,
                "motif_type": kind or "",
                "motif_members": "; ".join(motif_members.get(n, [])),
                "domain_notes": "; ".join(sorted(variant_meta[n]["domain_notes"])) if n in variant_meta else ""
            },
            "classes": f"L{layer} {role_class}" + (f" motif-{kind}" if kind else ""),
            "style": {"width": size, "height": size}
        }

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
            html.Div("Domain & Variant Information", 
                    style={'fontSize': 18, 'fontWeight': 'bold', 
                           'marginBottom': 15, 'color': '#2c3e50',
                           'borderBottom': '2px solid #ecf0f1',
                           'paddingBottom': 10}),
            html.Div("Click on an edge to see detailed information about domains, variants, and clinical data.",
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
                    'font-size': 18, 'font-weight': 'bold',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center'}},
                # Layer semantics: L0 (protein+variants), L1 (intermediate), L2 (endpoint)
                {'selector': '.L0',
                 'style': {'background-color': '#aacdd7',
                          'color': '#004466'}},
                {'selector': '.L1', 
                 'style': {'background-color':"#8bb6b3",
                           'color': '#125652'}},
                {'selector': '.L2',
                 'style': {'background-color': '#fabf77',
                          'color': '#b05e04'}},
                # Role classes override layer defaults for readability.
                {'selector': '.role-protein',
                 'style': {'background-color': '#aacdd7', 'color': '#004466'}},
                {'selector': '.role-variant',
                 'style': {'background-color': '#a492bb', 'color': '#573d82'}},
                {'selector': '.role-intermediate',
                 'style': {'background-color': '#cce9b6', 'color': '#3f6330'}},
                {'selector': '.role-endpoint',
                 'style': {'background-color': '#fabf77', 'color': '#b05e04'}},
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

@app.callback(
    Output({'type': 'edge-info', 'prot': MATCH}, 'children'),
    Input({'type': 'cy-net', 'prot': MATCH}, 'tapEdgeData'),
    prevent_initial_call=True)
def show_edge_info(edge):
    if not edge:
        return [
            html.Div("Domain & Variant Information", 
                    style={'fontSize': 18, 'fontWeight': 'bold', 
                           'marginBottom': 15, 'color': '#2c3e50',
                           'borderBottom': '2px solid #ecf0f1',
                           'paddingBottom': 10}),
            html.Div("Click on an edge to see detailed information about domains, variants, and clinical data.",
                    style={'color': '#7f8c8d', 'fontSize': 14, 'lineHeight': '1.4'})
        ]

    content = []

    content.append(
        html.Div("Edge Information", 
                style={'fontSize': 18, 'fontWeight': 'bold', 
                       'marginBottom': 15, 'color': '#2c3e50',
                       'borderBottom': '2px solid #ecf0f1',
                       'paddingBottom': 10})
    )

    rel = edge.get('rel', 'N/A')
    source = edge.get('source', 'N/A')
    target = edge.get('target', 'N/A')

    content.append(
        html.Div([
            html.Div("Relationship", 
                    style={'fontSize': 14, 'fontWeight': 'bold', 
                        'color': '#34495e', 'marginBottom': 5}),
            html.Div(f"{source} → {target}", 
                    style={'fontSize': 14, 'color': '#2c3e50', 
                        'marginBottom': 8, 'fontWeight': 'bold'}),
            *( [] if rel in ['DV', 'PV', 'has_domain'] else [
                html.Div(f"Type: {rel}", 
                    style={'fontSize': 14, 'color': '#7f8c8d'})
            ]),
            *( [] if not edge.get('evidence_count') else [
                html.Div(f"Evidence count: {edge['evidence_count']}",
                    style={'fontSize': 13, 'color': '#7f8c8d', 'marginTop': 4})
            ])
        ], style={
            'background': '#ffffff',
            'padding': 12,
            'borderRadius': 6,
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'marginBottom': 15,
            'border': '1px solid #dee2e6'
        })
    )

    if rel == 'DV':
        if 'note' in edge and edge['note']:
            content.append(
                html.Div([
                    html.Div("Domain Description", 
                            style={'fontSize': 14, 'fontWeight': 'bold', 
                                   'color': '#34495e', 'marginBottom': 8}),
                    html.Div(edge['note'], 
                            style={'fontSize': 14, 'color': '#2c3e50',
                                   'lineHeight': '1.4'})
                ], style={
                    'background': '#ffffff',
                    'padding': 12,
                    'borderRadius': 6,
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                    'marginBottom': 15,
                    'border': '1px solid #dee2e6'
                })
            )
    else:
        if 'note' in edge and edge['note']:
            content.append(
                html.Div([
                    html.Div("Description", 
                            style={'fontSize': 14, 'fontWeight': 'bold', 
                                   'color': '#34495e', 'marginBottom': 8}),
                    html.Div(edge['note'], 
                            style={'fontSize': 14, 'color': '#2c3e50',
                                   'lineHeight': '1.4'})
                ], style={
                    'background': '#ffffff',
                    'padding': 12,
                    'borderRadius': 6,
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                    'marginBottom': 15,
                    'border': '1px solid #dee2e6'
                })
            )

    if 'clinvar_data' in edge and edge['clinvar_data']:
        data = edge['clinvar_data']
        content.append(
            html.Div([
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
            ], style={
                'background': '#ffffff',
                'padding': 12,
                'borderRadius': 6,
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                'marginBottom': 15,
                'border': '1px solid #dee2e6'
            })
        )

    if 'pmid' in edge and edge['pmid']:
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
        content.append(
            html.Div([
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
            ], style={
                'background': '#ffffff',
                'padding': 12,
                'borderRadius': 6,
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                'marginBottom': 15,
                'border': '1px solid #dee2e6'
            })
        )

    return content

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
