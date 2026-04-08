# FEBio Simulation Workflow

## 1. What This Module Is

This FEBio integration is a phase-1 research prototype for controlled ECM decision support.

It is intentionally limited:

- only fixed template-driven scenarios are supported
- the agent does not generate arbitrary `.feb` XML
- requests must pass schema validation before input generation

Current supported scenarios:

1. `bulk_mechanics`
2. `single_cell_contraction`
3. `organoid_spheroid`

## 2. Runtime Model

The execution chain is:

1. build a structured simulation request
2. validate request fields and ranges
3. generate `input.feb` from a fixed template
4. run FEBio CLI
5. parse output artifacts into `simulation_result.json`
6. derive decision metrics into `simulation_metrics.json`
7. write `final_summary.md`

This keeps the workflow reproducible, testable, and explainable.

## 3. Environment Configuration

Relevant environment variables:

```env
FEBIO_ENABLED=true
FEBIO_EXECUTABLE=/Applications/FEBioStudio/FEBioStudio.app/Contents/MacOS/febio4
FEBIO_TIMEOUT_SECONDS=300
FEBIO_DEFAULT_TMP_DIR=/path/to/tmp
```

Behavior:

- if `FEBIO_ENABLED=false`, the simulation workflow returns structured `unavailable`
- if `FEBIO_EXECUTABLE` is not set, the app tries to auto-discover `febio4` or `febio`
- if FEBio is not found, design still runs but reports that simulation verification was not performed

## 4. CLI Examples

### Bulk mechanics

```bash
python -m ecm_organoid_agent \
  --workflow simulation \
  --query "Bulk verify ECM candidate with FEBio" \
  --simulation-scenario bulk_mechanics \
  --target-stiffness 8 \
  --matrix-youngs-modulus 8 \
  --matrix-poisson-ratio 0.3
```

### Single-cell contraction

```bash
python -m ecm_organoid_agent \
  --workflow simulation \
  --query "Evaluate single-cell traction transmission in ECM" \
  --simulation-scenario single_cell_contraction \
  --target-stiffness 8 \
  --matrix-youngs-modulus 8 \
  --cell-contractility 0.02
```

### Organoid spheroid

```bash
python -m ecm_organoid_agent \
  --workflow simulation \
  --query "Evaluate organoid-ECM interaction with a spheroid proxy" \
  --simulation-scenario organoid_spheroid \
  --target-stiffness 8 \
  --matrix-youngs-modulus 8 \
  --organoid-radius 0.18
```

### Design + simulation

```bash
python -m ecm_organoid_agent \
  --workflow design \
  --query "Design a GelMA-like ECM near stiffness 8 Pa with FEBio verification" \
  --target-stiffness 8 \
  --design-run-simulation \
  --design-simulation-scenario bulk_mechanics \
  --design-simulation-top-k 2
```

## 5. Artifacts

For each simulation workflow run, the app writes:

```text
runs/<run_id>/simulation/
  input_request.json
  input.feb
  metadata.json
  febio_stdout.txt
  febio_stderr.txt
  runner_metadata.json
  simulation_result.json
  simulation_metrics.json
  final_summary.md
```

## 6. Common Failure Modes

### FEBio not installed

Symptoms:

- workflow status becomes `unavailable`
- report explicitly says simulation verification was not performed

Recommended action:

- install FEBio or set `FEBIO_EXECUTABLE`

### Timeout

Symptoms:

- runner status becomes `timed_out`
- stdout/stderr are still preserved

Recommended action:

- increase `FEBIO_TIMEOUT_SECONDS`
- reduce mesh size or time steps

### Missing output files

Symptoms:

- parser returns warnings such as missing logfile, plotfile, or restart dump

Recommended action:

- inspect `febio_stdout.txt` and `febio_stderr.txt`
- inspect `input.feb`
- verify the executable and working directory

### Design and FEBio disagree

Symptoms:

- mechanics-informed top candidate differs from FEBio-backed preferred candidate

Interpretation:

- this is not automatically a bug
- it indicates model discrepancy or candidate-to-simulation mapping uncertainty
- review both rankings and keep wet-lab validation in the loop

## 7. Scientific Boundary

This module is not:

- a general FE model authoring system
- a multi-physics platform
- a wet-lab protocol generator
- a substitute for experimental validation
