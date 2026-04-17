# VarAtlas

**VarAtlas** is an interactive web-based platform for exploring variant–gene–disease regulatory pathways derived from the [INDRA](https://indra.readthedocs.io/) knowledge graph. It provides two complementary views — a layered biological network graph and a protein-level variant map — enabling hypothesis-driven investigation of disease-associated genetic variants.

A live deployment is available at: [https://jicijiang-varatlas.hf.space](https://jicijiang-varatlas.hf.space)

---

## Features

- **Gene-centric and disease-centric network views** with crossing-minimised layered layout (Large Neighbourhood Search)
- **Variant map** — lollipop-style visualisation of mutation positions along the protein sequence
- **Statistics dashboard** — summary of variants, genes, biological processes, and diseases
- **ClinVar and dbSNP integration** — external links for each variant node
- **Box-drag zoom** on the variant map to subset the network to a specific amino acid window

---

## Project Structure

```
variants_network/
├── indra_variants/
│   ├── app/
│   │   ├── variant_network.py   # Main Dash application
│   │   ├── config.py            # Environment configuration
│   │   └── assets/              # Favicon and static assets
│   └── data/
│       └── variant_effect/
│           └── variant_effect/  # Per-gene TSV data files (2,561 genes)
├── Dockerfile                   # HuggingFace Spaces deployment
├── requirements.txt
└── README.md
```

---

## Running Locally

**Requirements:** Python 3.10+

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python3 -m indra_variants.app.variant_network
```

The app will be available at `http://localhost:8051` by default.

**Environment variables (optional):**

| Variable | Default | Description |
|----------|---------|-------------|
| `DASH_PORT` | `8051` | Port for the local server |
| `DASH_DEBUG` | `0` | Set to `1` to enable Dash debug mode |
| `DATA_DIR` | `indra_variants/data` | Path to the data directory |

---

## Data

Variant effect data is sourced from ClinVar and processed through the INDRA pipeline. Each TSV file corresponds to one gene and contains variant–disease associations with evidence metadata including PubMed IDs, ClinVar significance, and review star ratings.

---

## Dependencies

| Package | Version |
|---------|---------|
| dash | 3.0.3 |
| dash-bootstrap-components | 2.0.2 |
| dash-cytoscape | 1.0.2 |
| plotly | 5.24.1 |
| pandas | 2.1.4 |
| networkx | 3.2.1 |
| scipy | 1.13.1 |
| gunicorn | 23.0.0 |

