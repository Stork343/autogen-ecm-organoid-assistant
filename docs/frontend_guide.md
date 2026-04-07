# ECM Research Desk Guide

## 1. What This App Is

This frontend is a research control surface for the ECM mechanics design system. It is not a generic chat UI.

It supports four core research tasks:

1. Literature and evidence synthesis
2. Mechanics fitting from datasets
3. ECM inverse design from target mechanics
4. Batch campaign comparison across multiple stiffness windows

## 2. Recommended Workflow

### A. Use `design` when:

- you already know the target stiffness window
- you want a top-k shortlist of candidate ECM parameter sets
- you want a first-pass formulation template

### B. Use `design_campaign` when:

- you want to compare multiple target stiffness windows
- you want to see whether one material family can span a range
- you want a comparative decision table before experiments

### C. Use `mechanics` when:

- you already have creep / relaxation / elastic / frequency-sweep / cyclic data
- you want fitted moduli, viscosity, characteristic times, or loop / damping summaries
- you want model comparison, parameter intervals, and identifiability hints instead of one forced constitutive family

### D. Use `hybrid` when:

- you want literature, mechanics, and simulation in one report

## 3. How To Read The Main Outputs

### Physics Gate

- `solver_converged`: numerical solve reached the force-residual threshold
- `monotonicity_valid`: density / crosslink trends satisfy built-in physical checks
- `nonlinearity_valid`: stress-strain response shows strain stiffening
- `physics_valid`: all three conditions pass

### Design Metrics

- `stiffness_mean`: Monte Carlo mean bulk stiffness
- `risk_index`: variability penalty from Monte Carlo spread
- `feasible`: candidate satisfies hard constraints
- `stiffness_error`: distance from target stiffness

### Formulation Translation

This is a mechanics-informed recipe template, not a final validated experimental formulation.

Use it as:

1. a starting wet-lab anchor
2. a candidate family suggestion
3. a short list for local formulation screening

## 4. Files Written Per Run

### Design runs

- `design_validation.md`
- `design_agent.md`
- `design_sensitivity.md`
- `formulation_mapping.md`
- `design_summary.json`
- `final_summary.md`

### Campaign runs

- `campaign_validation.md`
- `campaign_agent.md`
- `formulation_mapping.md`
- `campaign_summary.json`
- `final_summary.md`

## 5. Practical Usage Notes

- Prefer constraints when you actually care about feasibility.
- Use `design_campaign` before committing to one target window.
- Compare formulation families in the Design Board before reading long reports.
- Use the report archive for narrative context and the JSON artifacts for structured downstream analysis.
- If concentration / curing are not enough, pass extra material-condition hints as JSON, such as temperature, initiator fraction, molecular weight, or degree of substitution.

## 6. Protected Public Access

If you expose this frontend beyond localhost, turn login protection on first.

Recommended `.env` settings:

```env
FRONTEND_REQUIRE_LOGIN=true
FRONTEND_USERNAME=researcher
FRONTEND_PASSWORD=strong_password_here
FRONTEND_PUBLIC_HOST=0.0.0.0
FRONTEND_PUBLIC_PORT=8525
```

Start the protected frontend with:

```bash
bash scripts/run_protected_frontend.sh
```

If you use Cloudflare Tunnel:

```bash
bash scripts/run_protected_frontend_cloudflare.sh
```

Then put it behind:

1. Cloudflare Tunnel + Cloudflare Access
2. Tailscale

Avoid exposing raw Streamlit directly to the open internet without an access layer.

## 7. Current Scientific Boundary

This system is already useful for mechanics-first ECM design.

It is not yet:

- a material-specific calibrated rheology platform
- a biology-mechanics co-optimization platform
- a substitute for wet-lab validation
