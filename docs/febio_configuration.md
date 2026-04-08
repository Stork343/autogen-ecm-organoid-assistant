# FEBio Configuration Guide

## 1. Goal

This guide explains how to connect the repository to a local FEBio installation without breaking the repository's controlled, template-driven safety model.

The app does not let the agent author arbitrary FEBio XML. FEBio is only used through validated simulation requests and fixed templates.

## 2. Required Environment Variables

Recommended `.env` fields:

```env
FEBIO_ENABLED=true
FEBIO_EXECUTABLE=/Applications/FEBioStudio/FEBioStudio.app/Contents/MacOS/febio4
FEBIO_TIMEOUT_SECONDS=300
FEBIO_DEFAULT_TMP_DIR=/path/to/febio_tmp
```

Meaning:

- `FEBIO_ENABLED`
  - turns FEBio integration on or off
- `FEBIO_EXECUTABLE`
  - points to the FEBio CLI binary
- `FEBIO_TIMEOUT_SECONDS`
  - controls the maximum runtime per simulation
- `FEBIO_DEFAULT_TMP_DIR`
  - controls the temporary workspace used by FEBio-related tasks

## 3. Auto-Discovery Behavior

If `FEBIO_EXECUTABLE` is not set, the code tries to discover:

1. `febio4`
2. `febio`

If neither is found:

- `simulation` workflow returns structured `unavailable`
- `design` still runs
- reports explicitly state that FEBio verification was not performed

## 4. Verify the Installation

### CLI-level check

```bash
which febio4
```

### Repository-level check

```bash
python -m ecm_organoid_agent \
  --workflow simulation \
  --query "Bulk verify ECM candidate with FEBio" \
  --simulation-scenario bulk_mechanics \
  --target-stiffness 8 \
  --matrix-youngs-modulus 8 \
  --matrix-poisson-ratio 0.3
```

Expected outputs:

- `runs/<run_id>/simulation/input.feb`
- `runs/<run_id>/simulation/simulation_result.json`
- `runs/<run_id>/simulation/simulation_metrics.json`
- `reports/<report_name>.md`

## 5. Recommended Local Setup

For workstation use:

```env
FEBIO_ENABLED=true
FEBIO_EXECUTABLE=/Applications/FEBioStudio/FEBioStudio.app/Contents/MacOS/febio4
FEBIO_TIMEOUT_SECONDS=300
```

For CI or no-FEBio environments:

```env
FEBIO_ENABLED=false
```

This keeps the simulation-facing tests deterministic and lets the repo degrade gracefully.

## 6. Safety Boundary

This repository intentionally keeps the FEBio boundary narrow:

- no arbitrary `.feb` generation by the agent
- no arbitrary shell execution path exposed through tools
- no uncontrolled mesh or boundary authoring path

All FEBio runs must originate from:

1. validated request schema
2. fixed scenario template
3. controlled runner invocation
