## NL VRP Planner

Turn a natural-language campaign goal into an executable multi-constraint vehicle routing plan. The pipeline:
- Strategic plan via Gemini → targeted campus queries
- Place discovery via Google Places (New) Text Search → task list with time/poster costs
- Travel-time matrix via Distance Matrix (with fallbacks)
- OR-Tools VRP solver with workday and poster-capacity constraints → per-team itineraries

### Requirements
- Python 3.9+
- APIs enabled in your Google Cloud project: Places API (New), Distance Matrix API
- Keys via environment variables: `GEMINI_API_KEY`, `GOOGLE_MAPS_API_KEY` (see `README_KEYS.md`)

Install dependencies:
```bash
pip install googlemaps google-generativeai ortools requests python-dotenv
```

### Quickstart
1) Create a `.env` file next to `nl_vrp_planner.py`:
```bash
GEMINI_API_KEY=your_gemini_key_here
GOOGLE_MAPS_API_KEY=your_maps_key_here
```
2) Run the planner:
```bash
python nl_vrp_planner.py
```

By default, the script contains a sample NYC university campaign request you can customize in `main()`.

### Outputs
- `outputs/run_YYYYMMDD_HHMMSS/vrp_itinerary.csv`: timestamped per-team route with poster usage
- `outputs/run_YYYYMMDD_HHMMSS/metadata.json`: captured config and detected travel policy

### Useful environment knobs (optional)
- `VRP_TRAVEL_POLICY`: `auto` (default), `auto_city`, `auto_campus`, `walking_only`, `transit_only`, `driving_only`
- `VRP_SOLVER_TIME_LIMIT_SEC`: solver time limit (default `120`)
- `VRP_WALK_MAX_METERS_CITY`, `VRP_WALK_MAX_METERS_CAMPUS`: thresholds for walking preference

See `nl_vrp_planner.py` for additional scaling and penalty parameters.

### Tests
```bash
python -m unittest -v
```

Notes: External API calls may incur costs. The script uses lightweight on-disk caching and graceful fallbacks to reduce quota usage.


