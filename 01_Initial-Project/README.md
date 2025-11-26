# Network Intrusion Detection (Binary Classification)

Goal: Classify network flows as benign vs malicious using realistic features (e.g., Zeek/Argus). Start with interpretable baselines and progress to stronger models.

Why it’s employable: Mirrors SOC workflows for triage, shows explainability for analyst trust, and demonstrates ingestion from pcap → features → serving.

Scope outline:
- Data: pcap → Zeek/Argus → features; small public subset for reproducibility
- Baselines: Logistic Regression + tree-based models; compare AUC/PR; calibration
- Explainability: SHAP for top drivers; model card with limitations
- Packaging: API/CLI, Docker, experiment tracking (MLflow), CI
- Observability: drift, data quality checks; synthetic traffic load
- Security: threat model, dependency scanning, safe configs

Stretch: multi-class attack family classification; near-real-time inference demo.


