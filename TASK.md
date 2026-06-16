# Weekly Task — 2026-06-12

**Project:** Network Monitoring & Traffic Analysis Lab  
**Type:** Code task  
**Tool:** Cursor

## Task

Get Zeek running on the homelab and verify end-to-end log flow into the ML pipeline. The classifier benchmark is complete (RF baseline at 95.1% / 0.992 AUC, production default set in `config.yaml`). The gap between "trained model" and "live detection system" is Zeek. This task closes it: install Zeek on a Proxmox LXC or directly on the OPNsense-adjacent host, configure it to capture traffic from the VLAN-segmented network, verify that the conn/http/dns log output parses correctly through the existing feature extraction pipeline, and confirm that `src/main.py infer` can consume a Zeek-derived flow CSV without errors.

## Deliverable

A Zeek instance running on the homelab producing conn.log output, a verified feature extraction run on at least one real Zeek capture (≥100 rows), and a `docs/experiments/2026-06-zeek-live-setup.md` documenting the install steps, the log schema mapping (Zeek field → pipeline feature), and any schema mismatches found between the UNSW-NB15 training data and real Zeek output. The doc becomes the reference for the next phase (Kafka streaming + live inference).

---

## How the pieces connect (read this first)

Nothing in this repo has been verified end-to-end on real Zeek traffic yet. Treat everything below as **prove it works**, not "it should work."

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
| `conn.log` | **Not supported directly** — convert to CSV first (see Phase 2C) |

**What `live` does today** (`python -m src.main live --interface <iface>`):

- Starts Zeek in the background on a network interface
- Polls `data/zeek_logs/conn.log` every 2 seconds
- Classifies new connections and logs alerts for `prediction == 1`

**Mapping code location:** `src/ingestion/pcap_to_features.py` (not `scripts/zeek_to_features.py` — that script does not exist yet; optional thin CLI wrapper if you want it).

**Model artifact:** `models/random_forest_baseline.joblib` (gitignored). You must train locally before inference works.

---

## Steps

Work top to bottom. Do not skip to homelab Zeek until Phase 0 passes on your dev machine.

### Phase 0 — Local smoke test (no Zeek, ~30 min)

Goal: confirm Python, model artifact, and `infer` work before adding Zeek complexity.

- [ ] **0.1** Clone repo and open a terminal in `Network-Monitoring/`
- [ ] **0.2** Create venv and install deps:
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # Linux:   source .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] **0.3** Confirm training data exists: `data/01_Raw/rawdata.csv`  
  If missing, you need the UNSW-NB15 subset you trained on (not in git — `*.csv` is gitignored).
- [ ] **0.4** Train the production model (creates `models/random_forest_baseline.joblib`):
  ```bash
  python -m src.main train --model random_forest_baseline
  ```
  Expected: no errors; artifact at `models/random_forest_baseline.joblib`.
- [ ] **0.5** Baseline inference on held-out enriched data (no Zeek):
  ```bash
  python -m src.main infer --input data/03_Enriched/X_test.csv --output reports/smoke_test.csv
  ```
  Note: `X_test.csv` is preprocessed (one-hot encoded). This mainly proves the artifact loads and `predict()` runs.  
  If you only have raw CSV, use a slice of `data/01_Raw/rawdata.csv` instead:
  ```bash
  python -m src.main infer --input data/01_Raw/rawdata.csv --output reports/smoke_test.csv
  ```
- [ ] **0.6** Check outputs:
  - `reports/smoke_test.csv` exists with `prediction`, `confidence` columns
  - `reports/inference_results.ndjson` updated
  - Log line like `INFERENCE COMPLETE: X/Y flagged malicious`

**Phase 0 pass:** train + infer on CSV completes without traceback.

---

### Phase 1 — Install Zeek on your dev machine (~30 min)

Goal: confirm `zeek` binary works before homelab deployment. Do this on the same machine you'll run `infer` from (laptop or LXC).

- [ ] **1.1** Install Zeek:
  - **Linux (Ubuntu 22.04 — recommended for LXC):** https://zeek.org/get-zeek/
    ```bash
    sudo apt-get install curl gnupg2
    curl -fsSL https://download.zeek.org/zeek-key.asc | sudo gpg --dearmor -o /usr/share/keyrings/zeek-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/zeek-archive-keyring.gpg] http://download.zeek.org/deb stable main" | sudo tee /etc/apt/sources.list.d/zeek.list
    sudo apt-get update && sudo apt-get install zeek
    ```
  - **Windows:** WSL2 + Ubuntu steps above, or install on the homelab LXC only and SCP files back for inference on laptop.
- [ ] **1.2** Verify install:
  ```bash
  zeek --version
  which zeek   # Linux
  where zeek   # Windows
  ```
- [ ] **1.3** Point config at the binary if needed — edit `src/config/config.yaml`:
  ```yaml
  zeek:
    binary: zeek    # or full path, e.g. /opt/zeek/bin/zeek
  ```
- [ ] **1.4** Quick Zeek sanity check (offline PCAP):
  ```bash
  mkdir -p data/zeek_logs
  zeek -r path/to/sample.pcap
  ls conn.log
  head -20 conn.log
  ```
  You need any small PCAP (even a few seconds of your own traffic). Public sample PCAPs work for plumbing tests.

**Phase 1 pass:** `zeek --version` works and `conn.log` is produced from a PCAP.

---

### Phase 2 — Batch inference (Zeek → model)

Three paths, easiest first. Complete **2A** before homelab work; **2B** is the main integrated path; **2C** is what you'll use with homelab `conn.log` files.

#### Phase 2A — Test feature mapping in isolation (no model)

Goal: verify `conn.log` parses and produces sensible feature columns.

- [ ] **2A.1** Run Zeek on a PCAP into the project log dir:
  ```bash
  cd data/zeek_logs
  zeek -r /path/to/sample.pcap
  ```
- [ ] **2A.2** Inspect `conn.log` — look for `#fields` header and data rows (not just comments).
- [ ] **2A.3** Run mapping in Python (from repo root):
  ```bash
  python -c "
  from src.ingestion.pcap_to_features import parse_conn_log, zeek_connections_to_dataframe
  conns = parse_conn_log('data/zeek_logs/conn.log')
  print(f'parsed {len(conns)} connections')
  df = zeek_connections_to_dataframe(conns)
  print(df.shape)
  print(df[['dur','proto','service','state','spkts','dpkts','sbytes','dbytes']].head())
  df.to_csv('data/zeek_logs/zeek_features.csv', index=False)
  "
  ```
- [ ] **2A.4** Confirm `data/zeek_logs/zeek_features.csv` has ≥1 row and columns like `dur`, `proto`, `service`, `state`, `ct_srv_src`, etc.

**Phase 2A pass:** CSV written; no parse errors; row count matches conn.log.

#### Phase 2B — Integrated batch: PCAP → infer (one command)

Goal: prove the full `ingest_pcap` → `predict` path.

- [ ] **2B.1** From repo root:
  ```bash
  python -m src.main infer --input /path/to/sample.pcap --output reports/zeek_batch_test.csv
  ```
- [ ] **2B.2** Expected flow in logs:
  1. `Running Zeek on ...`
  2. `Parsed N connections from ...`
  3. `Built feature DataFrame: N rows, ... columns`
  4. `Running inference on N connections`
  5. `INFERENCE COMPLETE`
- [ ] **2B.3** Review `reports/zeek_batch_test.csv` — mostly `benign` on normal traffic is expected.

**Phase 2B pass:** PCAP inference completes end-to-end.

#### Phase 2C — Batch from existing conn.log (homelab path)

Goal: homelab Zeek writes `conn.log` → you infer on your laptop or the same host.

- [ ] **2C.1** Copy homelab `conn.log` to `data/zeek_logs/conn.log` (SCP, shared mount, etc.)
- [ ] **2C.2** Convert to features CSV (same Python as 2A.3) → `data/zeek_logs/zeek_features.csv`
- [ ] **2C.3** Run inference:
  ```bash
  python -m src.main infer --input data/zeek_logs/zeek_features.csv --output reports/homelab_batch.csv
  ```
- [ ] **2C.4** Need ≥100 rows for deliverable — capture longer or on a busier VLAN if count is low.

**Phase 2C pass:** real homelab `conn.log` → predictions CSV, ≥100 flows documented.

---

### Phase 3 — Live inference (dev machine or homelab)

Goal: prove `python -m src.main live` runs continuously. Requires a network interface Zeek can capture on.

**Prerequisites**

- Zeek installed on the capture host
- Interface with visible traffic (not loopback)
- On Linux: often need `sudo` for live capture
- Promiscuous mode if using a mirror/SPAN port

- [ ] **3.1** Find your interface name:
  ```bash
  # Linux
  ip link show
  # Windows (PowerShell)
  Get-NetAdapter
  ```
- [ ] **3.2** Short live test (60 seconds):
  ```bash
  python -m src.main live --interface eth0 --duration 60
  ```
  Replace `eth0` with your interface (`enp0s3`, `Ethernet`, etc.).
- [ ] **3.3** Generate traffic while it runs (browse, ping, SSH) so `conn.log` gets rows.
- [ ] **3.4** Expected behavior:
  - Log: `Starting live Zeek capture on interface '...'`
  - Log: `Zeek PID: ...`
  - Periodic: `New batch: N connections`
  - On flagged flows: `ALERT: N malicious connections in latest batch`
- [ ] **3.5** Stop with Ctrl+C if no `--duration`; Zeek process should terminate cleanly.

**Known limitation to watch:** live mode passes only *new* connections into `zeek_connections_to_dataframe`, so windowed `ct_*` features may differ from batch mode. Document if predictions look odd — fix is a follow-up code change, not a blocker for first live test.

**Phase 3 pass:** live command runs 60s+, processes at least one batch, exits without traceback.

---

### Phase 4 — Homelab deployment (OPNsense + Proxmox)

Goal: Zeek sees real VLAN traffic, not just loopback on a dev laptop.

- [ ] **4.1** Create Ubuntu 22.04 LXC on Proxmox (1–2 GB RAM is enough for Zeek sensor).
- [ ] **4.2** Install Zeek (Phase 1 steps) inside the LXC.
- [ ] **4.3** Network placement — pick one:
  - **Mirror/SPAN (best):** switch or OPNsense sends copy of traffic to LXC NIC; interface in promiscuous mode.
  - **TAP on trunk:** LXC NIC receives VLAN-tagged mirror of internal traffic.
  - **Zeek on capture host only:** run `zeek -i <iface>` on a machine that actually sees inter-VLAN flows (OPNsense alone does not give you `conn.log` for ML unless traffic is mirrored to a sensor).
- [ ] **4.4** OPNsense stays the firewall — do not replace it with Zeek. Zeek is passive observation only.
- [ ] **4.5** Capture 5–15 minutes of normal homelab traffic:
  ```bash
  zeek -i eth1   # or use Zeek's systemd/deploy scripts for production
  ```
- [ ] **4.6** Copy `conn.log` to dev machine → Phase 2C.
- [ ] **4.7** Optional: run Phase 3 live on the LXC if Python + model artifact live there too.

**Phase 4 pass:** ≥100 real homelab flows through infer; documented in experiment doc.

---

### Phase 5 — Document and note schema gaps

- [ ] **5.1** Create `docs/experiments/2026-06-zeek-live-setup.md` with:
  - Where Zeek runs (host, interface, mirror setup)
  - Zeek version (`zeek --version`)
  - Row count captured
  - Field mapping table: Zeek `conn.log` column → UNSW feature (reference `pcap_to_features.py`)
  - Features zero-filled because conn.log lacks them: `sttl`, `dttl`, `swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat`, `trans_depth`, `response_body_len`, FTP/HTTP counters
  - `service` / `conn_state` values seen in homelab vs training (`docs/Data.md`)
  - Whether batch vs live inference behaved differently
  - Screenshots or log snippets of successful `infer` output
- [ ] **5.2** Log anything broken (script errors, empty conn.log, permission denied, wrong interface) — that's valuable for the next fix pass.

---

## Troubleshooting checklist

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No artifact at models/...` | Model not trained | Phase 0.4 |
| `Zeek binary not found` | Zeek not installed or wrong path | Phase 1; update `config.yaml` `zeek.binary` |
| `Zeek did not produce conn.log` | Bad PCAP, Zeek crash, wrong cwd | Run `zeek -r file.pcap` manually in empty dir; read stderr |
| `Parsed 0 connections` | Empty capture or wrong log path | Check `conn.log` has data rows below `#fields` |
| `KeyError` / column mismatch on infer | Feature CSV missing columns | Re-run `zeek_connections_to_dataframe`; compare columns to `config.yaml` `schema` |
| Live mode: no batches | Wrong interface, no traffic, permissions | `sudo`; verify with `tcpdump -i eth0 -c 10` |
| Live mode: Zeek exits immediately | Interface down or not permitted | `ip link set eth0 up`; check capabilities |
| All predictions benign on homelab | Expected — mostly normal traffic + domain shift | Not a failure for plumbing test; note in doc |
| All predictions malicious | Schema mismatch or bad features | Inspect `zeek_features.csv` for NaNs, `-` values, absurd `rate` |

---

## Success criteria (summary)

| Milestone | Done when |
|-----------|-----------|
| **A — Pipeline works** | Phase 0 complete (train + CSV infer) |
| **B — Zeek batch works** | Phase 2B or 2C complete (PCAP or conn.log → predictions) |
| **C — Zeek live works** | Phase 3 complete (live runs ≥60s, ≥1 batch) |
| **D — Homelab real** | Phase 4 + 2C with ≥100 homelab flows |
| **E — Documented** | `docs/experiments/2026-06-zeek-live-setup.md` written |

**Minimum viable task complete:** A + B + D + E (live/C is stretch — valuable but not required to close the Zeek gap for batch homelab inference).

---

## After this task (not in scope now)

- Thin CLI: `scripts/zeek_to_features.py conn.log -o features.csv` wrapping existing mapping code
- Fix live mode `ct_*` window to use full connection history
- Kafka producer on Zeek host → consumer running `predict()`
- Drift check: compare homelab feature distributions vs training data
- Add `infer --input conn.log` support directly in `main.py`

---

## Context for Cursor

The repo is at `PROJECTS/Network-Monitoring/` (also https://github.com/BOYSABIO/Network-Monitoring). Inference uses `models/<model>.joblib` (train first) — not `data/03_Enriched/` directly. Input schema: UNSW-NB15-style flow CSV — see `docs/Data.md`. Zeek field mapping lives in `src/ingestion/pcap_to_features.py`. Production model: `random_forest_baseline` in `config.yaml`.
