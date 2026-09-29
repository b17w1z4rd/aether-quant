# Validation record

Checked on Python 3.12.14; exact numerical dependency versions are recorded in `demo-results.json`.

- `python -m unittest discover -s tests -v`: 13 tests passed.
- `python -m aether_quant demo`: successful full calibration, projection, hedge optimization and report generation.
- CSV import with the example positions reproduced the demo risk metrics within 1e-5.
- A three-expiry input exercised the minimum supported chain size and generated its dashboard.
- A negative hedge budget exited with a clear error before creating an output directory.
- The HTML contains 15 chart panels and 16 script blocks. All generated JavaScript passed `node --check`.
- The HTML loads no external script sources; Plotly is embedded.
- The static board was rendered from saved run arrays and visually inspected.

Interactive behavior has not been exercised in a real browser in this build environment. The static board is a separate rendering of the research outputs, not a browser screenshot.

The GitHub Actions workflow is configured for Python 3.10 and 3.12; it has not run before repository publication.
