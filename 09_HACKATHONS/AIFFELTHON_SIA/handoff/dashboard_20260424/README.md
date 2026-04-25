# Dashboard Handoff

## What To Use

- `sia_schedule_schema.v1.json`
  Dashboard data contract. Frontend should build against this first.

- `sia_schedule_example_20260316.json`
  Schema-compliant example payload. Best file for initial UI binding.

- `schedule_20260421_real_tri-mix.json`
  Real pipeline output from a live-style run. Useful to understand current backend shape.

- `random_march_probe_tri_mix_llm_weatherfix.json`
  Real backtest samples after historical weather fix. Useful for testing mixed EO/SAR rendering.

## Recommendation

Use this order:

1. Build UI against `sia_schedule_schema.v1.json`
2. Render using `sia_schedule_example_20260316.json`
3. Compare with current raw pipeline outputs to see what serializer work remains

## Important Notes

- Current pipeline raw outputs are **not yet identical** to `sia_schedule_schema.v1.json`
- Example schema file is the intended frontend contract
- Real outputs contain the actual data richness and edge cases
- Historical backtest now uses archive weather API for past dates

## Suggested Frontend Binding

- Header/meta: `meta`
- Threat list: `cities`
- Main timeline/cards: `shots`
- Bottom satellite cards: `satellite_allocation`
- Warning banner/icons: `diagnostics`

## Questions Backend/Frontend Should Align On

- `mode` enum naming: `backtest/live/forecast` vs current internal naming
- `tle.source` enum normalization
- `city_id` generation rule
- optional facility fields (`target.facility_name`, `target.coords`)

## Command Used For Recent Live-Style Output

```bash
python3 run_integrated_pipeline.py \
  --date 20260421 \
  --fetch-gdelt \
  --scenario tri-mix \
  --hours 168 \
  --use-llm \
  --refresh
```
