# Troubleshooting

## 1. FEBio Not Found

Symptoms:

- simulation workflow returns `unavailable`
- design report says FEBio verification was not performed

Checks:

```bash
which febio4
```

Fix:

- install FEBio
- or set `FEBIO_EXECUTABLE` in `.env`

## 2. Simulation Timed Out

Symptoms:

- runner status is `timed_out`
- `febio_stdout.txt` / `febio_stderr.txt` are written

Fix:

- increase `FEBIO_TIMEOUT_SECONDS`
- reduce `mesh_resolution`
- reduce `time_steps`

## 3. Missing `simulation_result.json` or `simulation_metrics.json`

Symptoms:

- report exists but structured simulation artifacts are incomplete

Checks:

- inspect `runs/<run_id>/simulation/febio_stdout.txt`
- inspect `runs/<run_id>/simulation/febio_stderr.txt`
- inspect `runs/<run_id>/simulation/input.feb`

Likely causes:

- FEBio executable failed before writing outputs
- malformed environment configuration
- unsupported local FEBio runtime behavior

## 4. Missing Log / Plot / Restart Files

The parser now records warnings when any of these are missing:

- logfile
- plotfile
- restart dump

Interpretation:

- missing plot or restart dump is not always fatal
- missing logfile weakens post-run interpretability and should be investigated

## 5. Design and Simulation Rankings Disagree

Symptoms:

- design shortlist favors one candidate
- FEBio-backed comparison favors another

Interpretation:

- this is a model discrepancy signal, not necessarily a bug
- the deterministic design search and FEBio templates are different abstractions

Recommended action:

- keep both the recommended and retained candidates
- review mapping assumptions in the design report
- decide whether wet-lab screening should cover both

## 6. Frontend Shows No Simulation Runs

Checks:

- confirm that `simulation` or `design --design-run-simulation` was executed
- confirm that `runs/<run_id>/metadata.json` has workflow `simulation` or that `design_summary.json` contains `design_simulation`

## 7. CI Without FEBio

This is expected.

Recommended setup:

```env
FEBIO_ENABLED=false
```

The test suite should still verify:

- command construction
- parser robustness
- metrics extraction
- graceful failure behavior
