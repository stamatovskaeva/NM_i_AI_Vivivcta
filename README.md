# NM_i_AI_Vivivcta

Top-level repository for multiple AI competition tasks.

This repository is organized by task folder (one folder per task), so root stays clean and each task is self-contained with its own code, scripts, requirements, models, and instructions.

## Repository structure

- [Astar](Astar)  
  World Championship AI task work (round strategy, analysis, replay tools, and task-specific scripts).

- [NorgesGruppen Data](NorgesGruppen%20Data)  
  Object detection competition task (training helpers, inference runtime, model weights, and submission-ready pipeline).

- [Tripletex](Tripletex)  
  Tripletex task work (main workflow script, scenario sweep utilities, and task-specific requirements).

## How to use this repository

1. Go into the task folder you are working on.
2. Follow that folder's local README/instructions.
3. Keep task-specific files inside that folder (not in root).

Each task folder should contain everything needed for that task workflow.

## Competition context

This repository contains separate deliverables for different competitions/challenges.  
Each task folder may have different:

- runtime requirements
- model files and artifacts
- scripts and workflows
- submission format

## For contributors / merge workflow

- Add new work under the correct task folder.
- Avoid mixing files from different tasks.
- Keep root-level files minimal and generic.

If a new task is started, create a new top-level folder and add it to this README.

## Quick start by task

- [Astar](Astar): start from [Astar/main.py](Astar/main.py), then check [Astar/requirements.txt](Astar/requirements.txt).
- [NorgesGruppen Data](NorgesGruppen%20Data): start from [NorgesGruppen Data/run.py](NorgesGruppen%20Data/run.py) and [NorgesGruppen Data/README.md](NorgesGruppen%20Data/README.md).
- [Tripletex](Tripletex): start from [Tripletex/main.py](Tripletex/main.py) and [Tripletex/README.md](Tripletex/README.md).
