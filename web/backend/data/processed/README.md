# Processed data

This folder contains the compact datasets used by the application.

Core traffic/model files:
- `road_catalog.csv`
- `road_hour_stats.csv.gz`
- `traffic_patterns.csv.gz`
- `traffic_training.csv.gz`
- `accident_road_hour_summary.csv.gz`
- `model_metrics.json`

Pro intelligence files:
- `segment_catalog.csv` — project Traffic `SegmentID` metadata used to request official NYC LION geometry in the browser.
- `accident_causes.csv.gz` — historical collision contributing factors by normalized road.
- `accident_time_profiles.csv` — peak crash window/day/weekend summaries.
- `government_priority.csv` — combined decision-support priority ranking.
