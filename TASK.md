# Weekly Task — 2026-06-12

**Project:** Network Monitoring & Traffic Analysis Lab
**Type:** Code task
**Tool:** Cursor

## Task

Get Zeek running on the homelab and verify end-to-end log flow into the ML pipeline. The classifier benchmark is complete (RF baseline at 95.1% / 0.992 AUC, production default set in `config.yaml`). The gap between "trained model" and "live detection system" is Zeek. This task closes it: install Zeek on a Proxmox LXC or directly on the OPNsense-adjacent host, configure it to capture traffic from the VLAN-segmented network, verify that the conn/http/dns log output parses correctly through the existing feature extraction pipeline, and confirm that `src/main.py infer` can consume a Zeek-derived flow CSV without errors.

## Deliverable

A Zeek instance running on the homelab producing conn.log output, a verified feature extraction run on at least one real Zeek capture (≥100 rows), and a `docs/experiments/2026-06-zeek-live-setup.md` documenting the install steps, the log schema mapping (Zeek field → pipeline feature), and any schema mismatches found between the UNSW-NB15 training data and real Zeek output. The doc becomes the reference for the next phase (Kafka streaming + live inference).

## Context for Cursor

The repo is at `PROJECTS/Network-Monitoring/` (also https://github.com/BOYSABIO/Network-Monitoring). The trained RF classifier reads from `data/03_Enriched/` via `src/main.py infer --input <path>`. The input schema comes from the UNSW-NB15-style flow CSV — see `docs/Data.md` for field definitions. Goal: (1) install Zeek on a Proxmox LXC (Ubuntu 22.04 recommended, Zeek from the official apt repo); (2) configure a capture interface pointing at the LAN/VLAN interface; (3) run a short capture and locate `conn.log`; (4) write a mapping script (`scripts/zeek_to_features.py`) that reads Zeek conn.log fields and outputs a CSV that matches the pipeline's expected input schema; (5) run `src/main.py infer` on the output and confirm predictions flow through without errors; (6) write the experiment doc. Key watch: Zeek's conn.log uses different field names than the UNSW-NB15 CSV — the mapping script is the core deliverable. Success condition: `python scripts/zeek_to_features.py conn.log | python -m src.main infer --input /dev/stdin` produces a predictions CSV.
