# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Notes: 
- Dates are in YYYY-MM-DD format.
- Higher priorities are greater in priority number. E.g., a priority 2 task is higher priority than a priority 1 task.

## [Unreleased]
### Added
- Added an option to the Model class to unload the model from memory.
- Added a CHANGELOG.md to track updates to the project.
- Added commandline arguments to set up hyperparameter sweeping using `hyperopt`.
- Added a commandline argument for setting the number of CNN layers to use.
- Added a commandline argument for setting the data directory.
- Added the Model class.

### Changed
- Updated `requirements.txt`.
- Delegated saving functions to the Model class.
- Switched code to Python 3.6. The code is probably no longer useable in Python<3.
- Moved `nb_layers` to the `hyperparameters` dict.
- The `hyperparameters` dict is now an OrderedDict (from collections).

### Deprecated
- Discontinued Docker support.

### Removed

### Fixed

### Security