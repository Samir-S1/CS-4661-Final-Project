# CS-4661 Final Project

A machine learning project for CS-4661 coursework.

## Project Overview

This repository contains the implementation and research for our CS-4661 (Machine Learning) final project. The project focuses on developing and analyzing machine learning models using Python and related data science tools.

## Repository Structure

```
├── README.md           # Project documentation
├── aqs_data_analysis/  # Air Quality System (AQS) data analysis and modeling
├── misc/               # Temporary workspace and experimental code (not part of final deliverable)
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Samir-S1/CS-4661-Final-Project.git
   cd CS-4661-Final-Project
   ```

2. Review the documentation in each subfolder for environment setup and usage instructions.

## Misc Directory

The `misc/` directory contains temporary experimental code, prototypes, and work-in-progress materials. This directory is frequently cleaned and updated, and its contents are not considered part of the final project deliverable.

## AQS Data Analysis

The `aqs_data_analysis/` folder contains scripts, notebooks, and documentation for the air quality data analysis component of the project. Refer to its README for details on data sources, setup, and analysis workflow.

## Prevent committing Jupyter notebook outputs

Large Jupyter notebook outputs (images, attachments, long printed tables) can bloat the repository. To avoid accidentally committing notebook outputs, install and enable a notebook-stripper such as `nbstripout` which automatically removes output cells when you commit.

Recommended steps (run in your project root):

PowerShell (Windows):

```powershell
python -m pip install --user nbstripout
python -m nbstripout --install
```

Git Bash / Linux / macOS:

```bash
python -m pip install --user nbstripout
python -m nbstripout --install
```

Notes:

- If you use a shared CI or have a project-wide policy, consider installing `nbstripout` globally or adding it to your repository's setup scripts so all contributors have it enabled.
- An alternative is to use Git LFS for large notebook attachments or large binary assets, but `nbstripout` is lightweight and prevents output history from being committed in the first place.

## Authors

- Samir S. (Samir-S1)
