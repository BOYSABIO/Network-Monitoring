# Network Monitoring & Traffic Analysis Lab

**Systems · Observability · Security Fundamentals · Data-Driven Analysis**

This repository documents a **personal network monitoring and observability lab** designed to build deep, hands-on understanding of **network traffic, visibility, and security fundamentals**.

The project focuses on **operating real infrastructure**, capturing and analyzing network data, and building a foundation for **security monitoring and future ML-based detection**.

---

## Project Goals

The primary objectives of this project are to:

- Build end-to-end **network visibility**
- Understand **traffic flows, protocols, and behavior**
- Implement **monitoring, logging, and observability**
- Develop security intuition through **hands-on system operation**
- Create a realistic foundation for **data-driven security analysis**

Rather than simulating attacks or solving isolated challenges, this lab emphasizes **continuous monitoring and system awareness**, similar to real-world environments.

---

## High-Level Architecture

At a high level, the lab consists of:

- Segmented internal network
- Firewall and routing layer
- Traffic capture and inspection
- Centralized logging and monitoring
- Analysis and visualization components

Traffic is observed passively to understand baseline behavior before introducing detection or analytics layers.

> ⚠️ Detailed configurations and sensitive information are intentionally excluded.

---

## Core Components

### Network & Systems
- Segmented network design
- Firewalling and traffic control
- Linux-based services and tooling

### Traffic Monitoring
- Packet capture and inspection
- Protocol-level visibility
- Flow-based analysis

### Logging & Observability
- Centralized logs
- Metrics collection
- System and network visibility

### Analysis
- Manual traffic analysis
- Pattern recognition and baselining
- Preparation for data-driven detection approaches

---

## Security Perspective

This project is **defensive-first** and focuses on:

- Understanding normal vs abnormal traffic
- Reducing blind spots in network visibility
- Learning how misconfigurations manifest in traffic
- Building intuition before automation

Rather than jumping straight to intrusion detection or ML models, the emphasis is on **foundational understanding** — a prerequisite for meaningful security analytics.

---

## CLI Usage

All pipeline commands are run through a single entry point. Requires dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Train a model

```bash
python -m src.main train --model logistic_regression
python -m src.main train --model random_forest
python -m src.main train --model logistic_regression --data path/to/data.csv
```

### Evaluate a trained model

```bash
python -m src.main evaluate
python -m src.main evaluate --model random_forest
```

### Run inference on a PCAP or CSV file

```bash
python -m src.main infer --input capture.pcap
python -m src.main infer --input data/test_data.csv
python -m src.main infer --input capture.pcap --output reports/results.csv
```

### Live monitoring on a network interface

```bash
python -m src.main live --interface eth0
python -m src.main live --interface wlan0 --duration 300
```

Requires [Zeek](https://zeek.org/get-zeek/) to be installed.

### OLS feature significance analysis

```bash
python -m src.main ols
python -m src.main ols --data path/to/data.csv --output reports/ols/
```

---

## Pipeline Architecture

The ML pipeline under `src/` follows this order:

```
config         Configuration loader (config.yaml)
   ↓
data_load      Load raw CSV data
   ↓
data_validation   Schema checks and data quality gates
   ↓
preprocess     Clean, encode, and transform features
   ↓
features       Feature engineering and selection
   ↓
model          Training with GridSearchCV, model registry
   ↓
evaluation     Metrics, confusion matrix, ROC curves
   ↓
inference      Classify new connections using a trained model
   ↓
ingestion      PCAP → Zeek → feature DataFrame (batch and live)
   ↓
api            REST API for serving predictions
```

Supporting modules: `utils` (logging, OLS analysis).

---

## Future Directions

Planned and potential extensions include:

- Traffic baselining and anomaly detection
- Feature extraction for ML-based monitoring
- Alerting and threshold-based detection
- Integration with additional data sources
- Visualization dashboards for operational insights

---

## Why This Project Exists

Many security and ML projects start with tools or models.  
This project starts with **visibility**.

By operating and observing a real network environment, the goal is to build intuition that can later support:
- Detection engineering
- Security analytics
- Infrastructure-aware ML systems
- Reliable and explainable monitoring solutions

---

## Notes

- This repository evolves over time as the lab grows
- Some configurations remain private for security reasons
- Documentation focuses on concepts, architecture, and learnings rather than copy-paste configs

