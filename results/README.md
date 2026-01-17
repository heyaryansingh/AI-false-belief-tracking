# Results Directory

This directory contains experiment results, metrics, figures, tables, and reports.

## Structure

- `metrics/` - Raw metrics CSV files
- `figures/` - Generated plots (PNG)
- `tables/` - Generated tables (Markdown)
- `manifests/` - Experiment manifests (JSON) with git hash, config hash, versions
- `reports/` - Generated technical reports (Markdown)

## Manifest Format

Each manifest includes:
- `git_hash`: Git commit hash
- `config_hash`: Hash of configuration files
- `timestamp`: Experiment timestamp
- `versions`: Package versions
- `config`: Experiment configuration
