---
layout: page
title: CIRCA
description: A research prototype for strict placement-level OOD co-location scheduling with counterfactual scoring and audited decision evaluation.
img: assets/img/project_preview/circa.png
img_no_responsive: true
importance: 2
category: open-source
status: Prototype
github: https://github.com/ShuhongDai/CIRCA
---

CIRCA is a research prototype for **strict placement-level OOD co-location scheduling** in multi-tenant GPU systems.
It studies how to score candidate placements under genuinely unseen deployment contexts, and how to evaluate whether a scheduler can still rank actions correctly when the placement structure was never observed during training.

## What It Does

- Scores candidate placements with a counterfactual decision-oriented view under strict OOD settings
- Provides a reproducible pipeline for benchmark collection, training, evaluation, and audit
- Includes audited main-result artifacts and representative failure-case extraction
- Focuses on decision quality for co-location scheduling rather than only fitting historical interference patterns

## Why It Is Interesting

In practical GPU scheduling, the key question is not only whether a predictor can output plausible interference scores, but whether it can make the **right placement decision** when the runtime context is genuinely new.
CIRCA is interesting because it emphasizes **decision evaluation under unseen placements**, which is closer to the real deployment challenge than standard in-distribution scoring.

## Repository Highlights

- Core package in `circa/` for data handling, benchmark utilities, and predictors
- Runnable scripts in `scripts/` for collection, training, evaluation, audit, and analysis
- Locked benchmark and audited result artifacts for reproducible evaluation
- Public research release under the MIT License

## Links

- Repository: [ShuhongDai/CIRCA](https://github.com/ShuhongDai/CIRCA)
- README: [Project overview and quick start](https://github.com/ShuhongDai/CIRCA/blob/main/README.md)
