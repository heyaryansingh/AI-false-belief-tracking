# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-17

### Added
- Initial release of belief-sensitive embodied assistance research repository
- GridHouse symbolic simulator with partial observability
- VirtualHome 3D simulator integration (optional)
- Three helper agent models: Reactive, Goal-Only, and Belief-Sensitive (particle filter)
- Comprehensive experiment framework with episode generation, evaluation, and analysis
- Automated metrics computation: false-belief detection, belief tracking, task performance, intervention quality
- Analysis pipeline: plots, tables, and technical report generation
- Full test suite (127+ tests) with 60% code coverage
- CI/CD pipeline with GitHub Actions
- Reproducibility features: deterministic seeding, manifest tracking, config hashing

### Features
- Dual simulator support (VirtualHome and GridHouse)
- Online belief tracking using particle filters
- False-belief detection and intervention policies
- Comprehensive evaluation metrics
- Automated analysis and reporting
- Full reproducibility pipeline
