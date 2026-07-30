# Network Monitoring — Context

**Project:** Network Monitoring & Traffic Analysis Lab  
**Repo:** https://github.com/BOYSABIO/Network-Monitoring

## What we are doing

Get Zeek running on the homelab and verify end-to-end log flow into the ML pipeline. The classifier benchmark is complete (RF baseline at 95.1% / 0.992 AUC, production default in `config.yaml`). The gap between "trained model" and "live detection system" is Zeek.

**Deliverable (when milestones close):** A Zeek instance producing `conn.log`, a verified feature extraction run on ≥100 real Zeek flows, and `docs/experiments/2026-06-zeek-live-setup.md` documenting install, Zeek→pipeline field mapping, and schema mismatches vs UNSW-NB15 training data. That doc is the reference for the next phase (Kafka streaming + live inference).

**Progress tracking:** GitHub is the source of truth — not this file.

- [Milestones](https://github.com/BOYSABIO/Network-Monitoring/milestones)
- [Open issues](https://github.com/BOYSABIO/Network-Monitoring/issues)

Work top to bottom: do not skip to homelab Zeek until M0 (local smoke test) passes on a dev machine. Nothing in this repo has been verified end-to-end on real Zeek traffic yet — treat plumbing as **prove it works**, not "it should work."

---

## How the pieces connect

```
                    ┌─────────────────────────────────────────┐
  Network traffic   │  Zeek (external — not Python)           │
  or PCAP file  ──► │  produces conn.log (tab-separated)      │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  src/ingestion/pcap_to_features.py      │
                    │  parse_conn_log → zeek_connections_     │
                    │  to_dataframe (Zeek → UNSW-NB15 cols)   │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  src/inference/predictor.py             │
                    │  encode + scale + Random Forest         │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  reports/inference_results.csv        │
                    │  reports/inference_results.ndjson     │
                    └───────────────────────────────────────┘
```

**What `infer` accepts today** (`src/main.py`):

| Input | What happens |
|-------|----------------|
| `.pcap` / `.pcapng` | Runs Zeek → parses `conn.log` → features → model |
| `.csv` | Loads CSV directly (must have UNSW-NB15-style feature columns) |
| `conn.log` | **Not supported directly** — convert to CSV first (see M4 / issue for 2C path; backlog for direct support) |

**What `live` does today** (`python -m src.main live --interface <iface>`):

- Starts Zeek in the background on a network interface
- Polls `data/zeek_logs/conn.log` every 2 seconds
- Classifies new connections and logs alerts for `prediction == 1`

**Known live limitation:** only *new* connections go into `zeek_connections_to_dataframe`, so windowed `ct_*` features may differ from batch mode. Document during live tests; fix is a backlog issue.

**Mapping code:** `src/ingestion/pcap_to_features.py` (not `scripts/zeek_to_features.py` — that CLI does not exist yet; backlog).

**Model artifact:** `models/random_forest_baseline.joblib` (gitignored). Train locally before inference works. Production model name: `random_forest_baseline` in `config.yaml`. Input schema: UNSW-NB15-style flow CSV — see `docs/Data.md`.

---

## Success criteria → milestones

| Criterion | Done when | Milestone |
|-----------|-----------|-----------|
| **A — Pipeline works** | Train + CSV infer | [M0 — Local pipeline smoke test](https://github.com/BOYSABIO/Network-Monitoring/milestone/1) |
| **B — Zeek batch works** | PCAP or mapped conn.log → predictions (dev 2B) | [M2 — Zeek batch inference (dev)](https://github.com/BOYSABIO/Network-Monitoring/milestone/3) |
| **C — Zeek live works** | Live ≥60s, ≥1 batch | [M3 — Live inference (stretch)](https://github.com/BOYSABIO/Network-Monitoring/milestone/4) |
| **D — Homelab real** | ≥100 homelab flows through infer | [M4 — Homelab Zeek deployment](https://github.com/BOYSABIO/Network-Monitoring/milestone/5) |
| **E — Documented** | `docs/experiments/2026-06-zeek-live-setup.md` | [M5 — Experiment documentation](https://github.com/BOYSABIO/Network-Monitoring/milestone/6) |

**MVP complete:** A + B + D + E. Live (C) is stretch. Prerequisites: [M1 — Dev Zeek install](https://github.com/BOYSABIO/Network-Monitoring/milestone/2) before M2/M3.

**After MVP:** [Backlog — After Zeek gap closed](https://github.com/BOYSABIO/Network-Monitoring/milestone/7) — CLI wrapper, live `ct_*` fix, Kafka, drift check, `infer --input conn.log`.

---

## Troubleshooting checklist

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No artifact at models/...` | Model not trained | M0 — train RF baseline |
| `Zeek binary not found` | Zeek not installed or wrong path | M1; update `config.yaml` `zeek.binary` |
| `Zeek did not produce conn.log` | Bad PCAP, Zeek crash, wrong cwd | Run `zeek -r file.pcap` manually in empty dir; read stderr |
| `Parsed 0 connections` | Empty capture or wrong log path | Check `conn.log` has data rows below `#fields` |
| `KeyError` / column mismatch on infer | Feature CSV missing columns | Re-run `zeek_connections_to_dataframe`; compare columns to `config.yaml` `schema` |
| Live mode: no batches | Wrong interface, no traffic, permissions | `sudo`; verify with `tcpdump -i eth0 -c 10` |
| Live mode: Zeek exits immediately | Interface down or not permitted | `ip link set eth0 up`; check capabilities |
| All predictions benign on homelab | Expected — mostly normal traffic + domain shift | Not a failure for plumbing test; note in experiment doc |
| All predictions malicious | Schema mismatch or bad features | Inspect `zeek_features.csv` for NaNs, `-` values, absurd `rate` |

---

## Context for Cursor

Inference uses `models/<model>.joblib` (train first) — not `data/03_Enriched/` directly. Zeek field mapping lives in `src/ingestion/pcap_to_features.py`. Homelab path: OPNsense stays the firewall; Zeek is passive observation only (mirror/SPAN or equivalent to the sensor). Track work via GitHub milestones/issues; keep this file as architecture and constraints only.
