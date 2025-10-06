# This advanced script uses a powerful generative model (Gemini 2.5 Pro) to devise a
# campaign strategy from a high-level goal, then executes that strategy by planning
# an optimized VRP for multiple teams with multiple constraints (time and posters).
#
# Architectural Stages:
# 1. Strategic Plan Generation: Gemini 2.5 Pro interprets a user's goal, identifies
#    target universities, and defines specific search queries for high-traffic areas.
# 2. Location Discovery: Google Places API executes the AI's search queries to find spots.
# 3. Task Generation: Discovered spots are bundled into tasks with time and poster costs.
# 4. Multi-Constraint VRP Solving: OR-Tools solves for the optimal routes, balancing
#    workday duration and each team's poster capacity.

import google.generativeai as genai
import googlemaps
import json
import requests
import csv
from datetime import datetime
from pathlib import Path
import math
import os
import time
import hashlib
import random
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Best-effort .env support without hard dependency
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

############################################################################
# ----- 1. CONFIGURATION: EDIT API KEYS AND PROMPT -----
############################################################################

# --- API Keys ---
# You must enable: "Places API" and "Generative Language API". "Distance Matrix API" is optional
# and only used when VRP_USE_DISTANCE_MATRIX=1.
# Provide via environment variables (preferably using a local .env for development)
#   - GOOGLE_MAPS_API_KEY: for Places (New) API and optional Distance Matrix
#   - GEMINI_API_KEY: for Generative Language API (Gemini)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# --- System Prompt for Gemini Pro ---
# This prompt instructs the powerful LLM to act as a campaign strategist.
GEMINI_SYSTEM_PROMPT = """
You are a world-class campaign strategist for guerilla marketing. Your task is to take a high-level user goal and devise a concrete, actionable plan for a poster campaign.

Based on the user's request, you must:
1.  **Identify Key Constraints**: Extract the number of teams, their work duration in hours, and the number of posters each team carries.
2.  **Formulate a Strategy**: Interpret the user's goal (e.g., "hit high traffic areas," "presence on most campuses"). Based on your knowledge of the target, identify the most impactful universities and areas to target to achieve this goal.
3.  **Return a Structured Plan**: Output a JSON object with the following schema:
    - "plan_summary": A brief, one-sentence summary of your strategic approach.
    - "constraints": An object containing the extracted constraints:
        - "num_teams": An integer.
        - "workday_hours": An integer.
        - "posters_per_team": An integer.
        - "depot_address": A string for the start/end location (default to a central spot where the user wants to run the campaign if not specified).
    - "search_targets": A list of specific, targeted strings for the Google Places API to search. These should be more descriptive than just a university name (e.g., "main student library at Columbia University Morningside Heights", "cafes around NYU Washington Square Park", "bulletin boards in the CUNY Graduate Center").
"""

# --- Travel / Heuristic Configuration (env-overridable) ---
# VRP_TRAVEL_POLICY: route mode policy ('auto', 'auto_city', 'auto_campus', 'walking_only', 'transit_only', 'driving_only')
TRAVEL_POLICY = os.getenv('VRP_TRAVEL_POLICY', 'auto').strip().lower()

# VRP_WALK_MAX_METERS_CITY: max straight-line distance to prefer walking in city mode
WALK_MAX_METERS_CITY = int(os.getenv('VRP_WALK_MAX_METERS_CITY', '1200'))
# VRP_WALK_MAX_METERS_CAMPUS: max straight-line distance to prefer walking in campus mode
WALK_MAX_METERS_CAMPUS = int(os.getenv('VRP_WALK_MAX_METERS_CAMPUS', '2500'))

# VRP_WALK_SPEED_MPS: walking speed (m/s) used in haversine fallback
WALK_SPEED_MPS = float(os.getenv('VRP_WALK_SPEED_MPS', '1.4'))        # ~5.0 km/h
# VRP_TRANSIT_SPEED_MPS: transit average speed (m/s) used in haversine fallback
TRANSIT_SPEED_MPS = float(os.getenv('VRP_TRANSIT_SPEED_MPS', '7.5'))  # ~27 km/h avg
# VRP_DRIVE_SPEED_MPS: driving average speed (m/s) used in haversine fallback
DRIVE_SPEED_MPS = float(os.getenv('VRP_DRIVE_SPEED_MPS', '11.0'))     # ~40 km/h avg

# VRP_TRANSIT_OVERHEAD_SEC: fixed seconds added to transit fallback (wait/transfer)
TRANSIT_OVERHEAD_SECONDS = int(os.getenv('VRP_TRANSIT_OVERHEAD_SEC', '300'))
# VRP_DRIVE_OVERHEAD_SEC: fixed seconds added to driving fallback (parking etc.)
DRIVE_OVERHEAD_SECONDS = int(os.getenv('VRP_DRIVE_OVERHEAD_SEC', '120'))

# VRP_USE_DISTANCE_MATRIX: if true, use Google Distance Matrix API (costly). Defaults off.
USE_DISTANCE_MATRIX = os.getenv('VRP_USE_DISTANCE_MATRIX', '0').strip().lower() in ('1', 'true', 'yes', 'on')

# --- Dynamic poster/time scaling configuration ---
# VRP_POPULARITY_MAX: cap for review-count normalization (higher = less saturation)
POPULARITY_MAX = float(os.getenv('VRP_POPULARITY_MAX', '500'))
# VRP_POSTER_SCALE_BASE: base multiplier for poster allocation (type baseline)
POSTER_SCALE_BASE = float(os.getenv('VRP_POSTER_SCALE_BASE', '0.8'))
# VRP_POSTER_SCALE_POP_COEFF: weight of popularity in poster scaling
POSTER_SCALE_POP_COEFF = float(os.getenv('VRP_POSTER_SCALE_POP_COEFF', '0.6'))
# VRP_POSTER_SCALE_RATING_COEFF: weight of rating in poster scaling
POSTER_SCALE_RATING_COEFF = float(os.getenv('VRP_POSTER_SCALE_RATING_COEFF', '0.2'))
# VRP_RATING_BASELINE: rating threshold where scaling starts (> baseline increases)
RATING_BASELINE = float(os.getenv('VRP_RATING_BASELINE', '3.0'))
# VRP_RATING_RANGE: rating span over which rating contributes (e.g., 3..5 -> 2.0)
RATING_RANGE = float(os.getenv('VRP_RATING_RANGE', '2.0'))
# VRP_TIME_SCALE_BASE: base multiplier for on-site time scaling
TIME_SCALE_BASE = float(os.getenv('VRP_TIME_SCALE_BASE', '0.8'))
# VRP_TIME_POP_COEFF: weight of popularity in on-site time scaling
TIME_POP_COEFF = float(os.getenv('VRP_TIME_POP_COEFF', '0.6'))

# --- Penalty and solver knobs ---
# VRP_PRIORITY_PENALTY_WEIGHT: weight per unit priority in drop penalty
PRIORITY_PENALTY_WEIGHT = int(os.getenv('VRP_PRIORITY_PENALTY_WEIGHT', '100'))
# VRP_POSTER_PENALTY_WEIGHT: weight per poster in drop penalty
POSTER_PENALTY_WEIGHT = int(os.getenv('VRP_POSTER_PENALTY_WEIGHT', '50'))
# VRP_SOLVER_TIME_LIMIT_SEC: solver time limit seconds
SOLVER_TIME_LIMIT_SEC = int(os.getenv('VRP_SOLVER_TIME_LIMIT_SEC', '120'))
# VRP_SOLVER_RANDOM_SEED: seed for search randomization (0 = default)
SOLVER_RANDOM_SEED = int(os.getenv('VRP_SOLVER_RANDOM_SEED', '0'))
# VRP_FIRST_SOLUTION_STRATEGY: SAVINGS, PATH_CHEAPEST_ARC, etc.
FIRST_SOLUTION_STRATEGY = os.getenv('VRP_FIRST_SOLUTION_STRATEGY', 'SAVINGS').strip().upper()
# VRP_LOCAL_SEARCH_META: GUIDED_LOCAL_SEARCH, TABU_SEARCH, SIMULATED_ANNEALING, etc.
LOCAL_SEARCH_META = os.getenv('VRP_LOCAL_SEARCH_META', 'GUIDED_LOCAL_SEARCH').strip().upper()

# --- Lightweight disk cache & retry settings ---
# VRP_CACHE_DIR: directory for API response caches
CACHE_DIR = os.getenv('VRP_CACHE_DIR', str(Path(__file__).parent / '.cache'))
# VRP_CACHE_TTL_SEC: TTL for cached responses
CACHE_TTL_SEC = int(os.getenv('VRP_CACHE_TTL_SEC', '86400'))
# VRP_API_MAX_RETRIES: number of retries for external API calls
API_MAX_RETRIES = int(os.getenv('VRP_API_MAX_RETRIES', '2'))
# VRP_API_BACKOFF_BASE_MS: base backoff in ms (exponential)
API_BACKOFF_BASE_MS = int(os.getenv('VRP_API_BACKOFF_BASE_MS', '250'))

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def haversine_meters(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points (meters)."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def estimate_travel_seconds(distance_meters, mode):
    """Estimate travel seconds from straight-line distance using mode speeds."""
    if distance_meters is None:
        # Unknown distance; fall back to a conservative default
        return 1800
    speed = WALK_SPEED_MPS if mode == 'walking' else TRANSIT_SPEED_MPS if mode == 'transit' else DRIVE_SPEED_MPS
    # Add a small fixed overhead to account for stops/waits in transit/driving
    overhead = 0
    if mode == 'transit':
        overhead = TRANSIT_OVERHEAD_SECONDS
    elif mode == 'driving':
        overhead = DRIVE_OVERHEAD_SECONDS
    travel = distance_meters / max(speed, 0.1)
    return int(travel + overhead)

def choose_mode_for_distance(distance_meters, policy):
    if policy == 'walking_only':
        return 'walking'
    if policy == 'transit_only':
        return 'transit'
    if policy == 'driving_only':
        return 'driving'
    # Autos
    if policy == 'auto_campus':
        if distance_meters is None:
            return 'walking'
        return 'walking' if distance_meters <= WALK_MAX_METERS_CAMPUS else 'driving'
    # Default: auto_city
    if distance_meters is None:
        return 'transit'
    return 'walking' if distance_meters <= WALK_MAX_METERS_CITY else 'transit'

# --- Poster/time scaling helper (exposed for testing) ---
def compute_dynamic_poster_and_time(base_posters, base_time, rating, user_count):
    """Compute dynamic poster count and on-site time from base values and popularity.

    Returns a tuple: (dynamic_posters:int, dynamic_time_minutes:int)
    """
    try:
        rating_val = float(rating or 0)
    except Exception:
        rating_val = 0.0
    try:
        user_count_val = float(user_count or 0)
    except Exception:
        user_count_val = 0.0
    popularity = 0.0 if POPULARITY_MAX <= 0 else max(0.0, min(user_count_val, POPULARITY_MAX)) / POPULARITY_MAX
    rating_boost = 0.0 if RATING_RANGE <= 0 else max(0.0, min(rating_val, 5.0) - RATING_BASELINE) / RATING_RANGE
    poster_scale = POSTER_SCALE_BASE + POSTER_SCALE_POP_COEFF * popularity + POSTER_SCALE_RATING_COEFF * rating_boost
    dynamic_posters = max(1, int(round((base_posters or 0) * poster_scale)))
    if (base_time or 0) <= 0:
        dynamic_time = 0
    else:
        time_scale = TIME_SCALE_BASE + TIME_POP_COEFF * popularity
        dynamic_time = max(1, int(round((base_time or 0) * time_scale)))
    return dynamic_posters, dynamic_time

############################################################################
# ----- 2. PHASE 1: STRATEGIC PLAN GENERATION WITH GEMINI PRO -----
############################################################################

def generate_strategic_plan(user_request):
    """Uses the Gemini API to generate a strategic plan from a user's goal."""
    print("Generating strategic plan...")
    if not GEMINI_API_KEY:
        print("Missing Gemini API key.")
        return None
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a more powerful model for strategic reasoning
        model = genai.GenerativeModel(model_name='gemini-2.5-pro', system_instruction=GEMINI_SYSTEM_PROMPT)
        
        response = model.generate_content(user_request)
        
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        strategic_plan = json.loads(json_text)
        
        print(f"Plan: {strategic_plan['plan_summary']}")
        return strategic_plan
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

############################################################################
# ----- 3. PHASE 2 & 3: LOCATION DISCOVERY & TASK GENERATION -----
############################################################################

def places_text_search(api_key, text_query, page_size=20):
    """Use Places API (New) Text Search to find places for a free-form query.

    Returns a dict with a 'results' list to emulate legacy structure.
    """
    if not api_key:
        print("Missing GOOGLE_MAPS_API_KEY for Places search.")
        return {"results": []}
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # Request fields needed for ranking and addressing
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.location,places.types"
    }
    payload = {
        "textQuery": text_query,
        "pageSize": page_size
    }
    # Simple disk cache key
    key_raw = f"places|{text_query}|{page_size}"
    key = hashlib.sha256(key_raw.encode('utf-8')).hexdigest()
    cache_path = Path(CACHE_DIR) / f"{key}.json"
    now = time.time()
    try:
        if cache_path.exists() and (now - cache_path.stat().st_mtime) <= CACHE_TTL_SEC:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            return cached
    except Exception:
        pass

    # Live call with retries/backoff
    attempt = 0
    while attempt <= API_MAX_RETRIES:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            result = {"results": data.get("places", [])}
            try:
                with open(cache_path, 'w') as f:
                    json.dump(result, f)
            except Exception:
                pass
            return result
        except Exception as e:
            attempt += 1
            if attempt > API_MAX_RETRIES:
                print(f"Places search failed for '{text_query}': {e}")
                return {"results": []}
            backoff = (API_BACKOFF_BASE_MS / 1000.0) * (2 ** (attempt - 1)) * (1 + random.random())
            time.sleep(backoff)

def discover_and_generate_tasks(gmaps, strategic_plan):
    """
    Executes the AI's plan by using the Places API to find locations and generate tasks.
    """
    print("Discovering locations...")
    if not GOOGLE_MAPS_API_KEY:
        print("Missing GOOGLE_MAPS_API_KEY.")
        return None
    tasks = []
    
    depot_address = strategic_plan['constraints'].get('depot_address', 'Penn Station, NY')
    depot_lat = None
    depot_lng = None
    try:
        depot_geo = gmaps.geocode(depot_address)
        if depot_geo and 'geometry' in depot_geo[0] and 'location' in depot_geo[0]['geometry']:
            depot_lat = depot_geo[0]['geometry']['location'].get('lat')
            depot_lng = depot_geo[0]['geometry']['location'].get('lng')
    except Exception:
        pass
    tasks.append({
        'name': 'DEPOT', 
        'address': depot_address, 
        'priority': 100, 
        'time_on_site_min': 0,
        'poster_cost': 0,
        'lat': depot_lat,
        'lng': depot_lng
    })

    # Canonical categories and their costs
    location_type_config = {
        'library': {'priority': 80, 'time': 5, 'posters': 5},
        'student_center': {'priority': 75, 'time': 4, 'posters': 4},
        'bookstore': {'priority': 65, 'time': 3, 'posters': 3},
        'coffee_shop': {'priority': 45, 'time': 1, 'posters': 1},
        'dormitory': {'priority': 70, 'time': 3, 'posters': 3},
        'gym': {'priority': 55, 'time': 3, 'posters': 3},
        'bus_stop': {'priority': 35, 'time': 1, 'posters': 1},
        'transit_station': {'priority': 60, 'time': 4, 'posters': 4},
        'bulletin_board': {'priority': 70, 'time': 3, 'posters': 3},
        'dining_hall': {'priority': 65, 'time': 3, 'posters': 3},
        'health_center': {'priority': 60, 'time': 2, 'posters': 2},
        'recreation_center': {'priority': 55, 'time': 3, 'posters': 3},
        'athletics_venue': {'priority': 60, 'time': 4, 'posters': 4},
        'academic_building': {'priority': 50, 'time': 2, 'posters': 2},
        'student_services': {'priority': 55, 'time': 2, 'posters': 2},
        'bike_share_dock': {'priority': 35, 'time': 1, 'posters': 1},
        'ferry_terminal': {'priority': 55, 'time': 3, 'posters': 3},
        'grocery_store': {'priority': 50, 'time': 2, 'posters': 2},
        'pharmacy': {'priority': 50, 'time': 1, 'posters': 1},
        'laundromat': {'priority': 40, 'time': 2, 'posters': 2},
        'thrift_store': {'priority': 45, 'time': 2, 'posters': 2},
        'print_shop': {'priority': 55, 'time': 3, 'posters': 3},
        'community_center': {'priority': 60, 'time': 3, 'posters': 3},
        'park': {'priority': 45, 'time': 2, 'posters': 2},
        'plaza': {'priority': 45, 'time': 2, 'posters': 2},
        'entertainment_venue': {'priority': 50, 'time': 3, 'posters': 3},
        'movie_theater': {'priority': 50, 'time': 3, 'posters': 3},
        'bar_pub': {'priority': 40, 'time': 1, 'posters': 1},
        'parking_garage': {'priority': 30, 'time': 1, 'posters': 1},
        'campus_gate': {'priority': 30, 'time': 1, 'posters': 1},
        'museum': {'priority': 60, 'time': 3, 'posters': 3}
    }

    # Aliases mapping to canonical categories
    location_alias_map = {
        'library': 'library', 'libraries': 'library', 'public library': 'library', 'public libraries': 'library',
        'student center': 'student_center', 'student centers': 'student_center',
        'student union': 'student_center', 'student unions': 'student_center',
        'bookstore': 'bookstore', 'bookstores': 'bookstore',
        'cafe': 'coffee_shop', 'cafes': 'coffee_shop',
        'coffee shop': 'coffee_shop', 'coffee shops': 'coffee_shop',
        'cafeteria': 'coffee_shop', 'cafeterias': 'coffee_shop',
        'dining hall': 'dining_hall', 'dining halls': 'dining_hall',
        'food court': 'dining_hall', 'food courts': 'dining_hall',
        'canteen': 'dining_hall', 'canteens': 'dining_hall',
        'dormitory': 'dormitory', 'dormitories': 'dormitory', 'dorm': 'dormitory',
        'residence hall': 'dormitory', 'residence halls': 'dormitory',
        'gym': 'gym', 'gyms': 'gym',
        'bus stop': 'bus_stop', 'bus stops': 'bus_stop', 'shuttle stop': 'bus_stop', 'shuttle stops': 'bus_stop',
        'transit station': 'transit_station', 'transit stations': 'transit_station',
        'subway station': 'transit_station', 'subway stations': 'transit_station',
        'train station': 'transit_station', 'train stations': 'transit_station',
        'bus terminal': 'transit_station', 'bus terminals': 'transit_station',
        'ferry terminal': 'ferry_terminal', 'ferry terminals': 'ferry_terminal',
        'bike share': 'bike_share_dock', 'bike shares': 'bike_share_dock', 'bikeshare': 'bike_share_dock', 'bike-share': 'bike_share_dock', 'citibike': 'bike_share_dock', 'bike dock': 'bike_share_dock', 'bike docks': 'bike_share_dock',
        'student health center': 'health_center', 'student health centers': 'health_center', 'health center': 'health_center', 'health centers': 'health_center', 'counseling center': 'health_center', 'counseling centers': 'health_center',
        'recreation center': 'recreation_center', 'recreation centers': 'recreation_center',
        'athletic center': 'athletics_venue', 'athletics center': 'athletics_venue', 'stadium': 'athletics_venue', 'stadiums': 'athletics_venue', 'arena': 'athletics_venue', 'arenas': 'athletics_venue', 'intramural field': 'athletics_venue', 'intramural fields': 'athletics_venue',
        'academic building': 'academic_building', 'academic buildings': 'academic_building', 'study lounge': 'academic_building', 'study lounges': 'academic_building', 'computer lab': 'academic_building', 'computer labs': 'academic_building', 'makerspace': 'academic_building', 'makerspaces': 'academic_building',
        'career center': 'student_services', 'career centers': 'student_services', 'advising office': 'student_services', 'advising offices': 'student_services', 'registrar': 'student_services', 'bursar': 'student_services', 'copy center': 'student_services', 'copy centers': 'student_services', 'print center': 'student_services', 'print centers': 'student_services',
        'bulletin board': 'bulletin_board', 'bulletin boards': 'bulletin_board',
        'community center': 'community_center', 'community centers': 'community_center',
        'park': 'park', 'parks': 'park',
        'plaza': 'plaza', 'plazas': 'plaza', 'square': 'plaza', 'squares': 'plaza',
        'music venue': 'entertainment_venue', 'music venues': 'entertainment_venue', 'event space': 'entertainment_venue', 'event spaces': 'entertainment_venue', 'concert hall': 'entertainment_venue', 'concert halls': 'entertainment_venue',
        'movie theater': 'movie_theater', 'movie theaters': 'movie_theater', 'cinema': 'movie_theater', 'cinemas': 'movie_theater',
        'bar': 'bar_pub', 'bars': 'bar_pub', 'pub': 'bar_pub', 'pubs': 'bar_pub', 'tavern': 'bar_pub', 'taverns': 'bar_pub',
        'grocery store': 'grocery_store', 'grocery stores': 'grocery_store', 'supermarket': 'grocery_store', 'supermarkets': 'grocery_store', 'bodega': 'grocery_store', 'bodegas': 'grocery_store',
        'pharmacy': 'pharmacy', 'pharmacies': 'pharmacy', 'drug store': 'pharmacy', 'drug stores': 'pharmacy', 'drugstore': 'pharmacy', 'drugstores': 'pharmacy',
        'laundromat': 'laundromat', 'laundromats': 'laundromat', 'laundry': 'laundromat', 'laundries': 'laundromat',
        'thrift store': 'thrift_store', 'thrift stores': 'thrift_store', 'secondhand store': 'thrift_store', 'secondhand stores': 'thrift_store',
        'print shop': 'print_shop', 'print shops': 'print_shop', 'copy shop': 'print_shop', 'copy shops': 'print_shop',
        'parking garage': 'parking_garage', 'parking garages': 'parking_garage', 'parking lot': 'parking_garage', 'parking lots': 'parking_garage',
        'campus gate': 'campus_gate', 'campus gates': 'campus_gate', 'main gate': 'campus_gate', 'entrance gate': 'campus_gate', 'entrance gates': 'campus_gate',
        'museum': 'museum', 'museums': 'museum'
    }

    # Group places by a logical area (e.g., "Columbia University") and build per-place tasks
    campus_tasks = {}
    # Track total selected places to avoid huge matrices
    MAX_TASKS_PER_CAMPUS = 12
    MAX_TASKS_PER_CATEGORY_PER_CAMPUS = 4
    MAX_TOTAL_TASKS = 60
    total_selected = 0
    # Global dedupe across campuses/categories
    global_seen_ids = set()
    global_seen_addresses = set()
    # Track seen place names per campus to avoid duplicate counting
    campus_seen = {}

    for search_query in strategic_plan['search_targets']:
        print(f"Searching: {search_query}")
        try:
            # Identify the general campus area for grouping
            campus_key = search_query.split(' at ')[-1].split(' near ')[-1]
            if not campus_key:
                campus_key = search_query

            if campus_key not in campus_tasks:
                geocode_result = gmaps.geocode(campus_key)
                if not geocode_result: continue
                campus_tasks[campus_key] = {
                    'time': 0, 'posters': 0, 'priority_sum': 0, 'count': 0,
                    'address': geocode_result[0]['formatted_address'],
                    'category_counts': {},
                    'selected_count': 0,
                    'places': []
                }
                campus_seen[campus_key] = set()

            # Build augmented queries to broaden coverage around the campus
            augmented_terms = [
                'coffee shops', 'cafes', 'dormitories', 'student centers', 'libraries', 'bookstores',
                'dining halls', 'food courts', 'student health centers', 'recreation centers',
                'athletic centers', 'stadiums', 'grocery stores', 'pharmacies', 'laundromats',
                'print shops', 'bus stops', 'transit stations', 'shuttle stops', 'bike share docks',
                'community centers', 'parks', 'plazas', 'music venues', 'movie theaters',
                'bars', 'parking garages', 'campus gates', 'museums', 'bulletin boards'
            ]
            # Limit the number of extra calls per campus to avoid excessive API usage
            augmented_queries = [search_query] + [f"{term} near {campus_key}" for term in augmented_terms][:20]

            for q in augmented_queries:
                places_result = places_text_search(GOOGLE_MAPS_API_KEY, q)
                for place in places_result.get('results', []):
                    # Determine location type from search query to assign costs
                    q_lower = q.lower()
                    found_type = None
                    for alias, canonical in location_alias_map.items():
                        if alias in q_lower:
                            found_type = canonical
                            break
                    if not found_type:
                        found_type = 'coffee_shop'

                    # Deduplicate places by ID/address globally and by name per campus, then cap per category
                    place_id = place.get('id') or place.get('placeId') or ''
                    addr_text = (place.get('formattedAddress') or '').strip().lower()
                    if (place_id and place_id in global_seen_ids) or (addr_text and addr_text in global_seen_addresses):
                        continue
                    display_name = place.get('displayName')
                    if isinstance(display_name, dict):
                        name_text = (display_name.get('text') or '').strip()
                    else:
                        name_text = (display_name or '').strip()
                    if not name_text:
                        name_text = (place.get('formattedAddress') or '').strip()
                    norm_name = name_text.lower()

                    if norm_name in campus_seen[campus_key]:
                        continue

                    category_counts = campus_tasks[campus_key].setdefault('category_counts', {})
                    # Enforce per-category and per-campus caps, and a global cap
                    if category_counts.get(found_type, 0) >= MAX_TASKS_PER_CATEGORY_PER_CAMPUS:
                        continue
                    if campus_tasks[campus_key]['selected_count'] >= MAX_TASKS_PER_CAMPUS:
                        continue
                    if total_selected >= MAX_TOTAL_TASKS:
                        continue

                    # Accept this place
                    if place_id:
                        global_seen_ids.add(place_id)
                    if addr_text:
                        global_seen_addresses.add(addr_text)
                    campus_seen[campus_key].add(norm_name)
                    category_counts[found_type] = category_counts.get(found_type, 0) + 1
                    campus_tasks[campus_key]['selected_count'] += 1
                    total_selected += 1

                    # Compute a simple relevance score for ranking
                    rating = place.get('rating') or 0
                    user_count = place.get('userRatingCount') or place.get('userRatingsTotal') or 0
                    try:
                        score = float(rating) * float(user_count)
                    except Exception:
                        score = 0.0

                    # Dynamic poster/time model based on popularity and rating
                    config = location_type_config[found_type]
                    dynamic_posters, dynamic_time = compute_dynamic_poster_and_time(
                        config['posters'],
                        config['time'],
                        rating,
                        user_count
                    )

                    loc_obj = place.get('location') or {}
                    lat = loc_obj.get('latitude')
                    lng = loc_obj.get('longitude')

                    campus_tasks[campus_key]['places'].append({
                        'name': name_text if name_text else f"{found_type.title()} near {campus_key}",
                        'address': (place.get('formattedAddress') or campus_tasks[campus_key]['address']),
                        'priority': config['priority'],
                        'time_on_site_min': dynamic_time,
                        'poster_cost': dynamic_posters,
                        'score': score,
                        'lat': lat,
                        'lng': lng
                    })
                

        except Exception as e:
            print(f"Query error '{search_query}': {e}")

    # Convert the grouped tasks into the final list for the solver
    for campus, data in campus_tasks.items():
        if data.get('places'):
            # Sort selected places by score (desc) and append as individual tasks
            for p in sorted(data['places'], key=lambda x: x.get('score', 0), reverse=True):
                tasks.append({
                    'name': p['name'],
                    'address': p['address'],
                    'priority': p['priority'],
                    'time_on_site_min': p['time_on_site_min'],
                    'poster_cost': p['poster_cost'],
                    'lat': p.get('lat'),
                    'lng': p.get('lng')
                })
                print(f"Task: {p['name']} (time {p['time_on_site_min']}m, posters {p['poster_cost']}, prio {p['priority']})")

    if len(tasks) <= 1:
        print("No tasks generated.")
        return None
        
    return tasks

############################################################################
# ----- 4. PHASE 4: MULTI-CONSTRAINT VRP SOLVER -----
############################################################################

def create_data_model(tasks, constraints):
    """Stores the data for the problem, including poster constraints."""
    data = {}
    data['locations'] = [task['address'] for task in tasks]
    data['time_on_site'] = [task['time_on_site_min'] * 60 for task in tasks]
    data['priorities'] = [task['priority'] for task in tasks]
    data['task_names'] = [task['name'] for task in tasks]
    data['poster_costs'] = [task['poster_cost'] for task in tasks]
    data['latitudes'] = [task.get('lat') for task in tasks]
    data['longitudes'] = [task.get('lng') for task in tasks]
    
    data['num_vehicles'] = constraints['num_teams']
    data['depot'] = 0 
    data['workday_seconds'] = constraints['workday_hours'] * 3600
    data['poster_capacities'] = [constraints['posters_per_team']] * constraints['num_teams']
    return data

def build_time_matrix(gmaps, data):
    # Builds the travel time matrix with per-pair mode selection and haversine fallback
    addresses = data['locations']
    lats = data.get('latitudes') or []
    lngs = data.get('longitudes') or []
    n = len(addresses)
    try:
        if n == 0:
            return []

        # Determine effective policy automatically if requested
        def detect_effective_policy():
            if TRAVEL_POLICY != 'auto':
                return TRAVEL_POLICY
            # Use median non-depot pairwise straight-line distance to detect campus-like clustering
            distances = []
            for i in range(1, n):
                for j in range(i + 1, n):
                    di = haversine_meters(lats[i], lngs[i], lats[j], lngs[j])
                    if di is not None and di > 0:
                        distances.append(di)
            if not distances:
                return 'auto_city'
            distances.sort()
            median = distances[len(distances)//2]
            return 'auto_campus' if median <= WALK_MAX_METERS_CAMPUS else 'auto_city'

        effective_policy = detect_effective_policy()

        # Precompute straight-line distances and chosen modes per pair
        dist_matrix = [[None for _ in range(n)] for _ in range(n)]
        mode_matrix = [[None for _ in range(n)] for _ in range(n)]
        modes_needed = set()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dij = haversine_meters(lats[i], lngs[i], lats[j], lngs[j])
                dist_matrix[i][j] = dij
                mode = choose_mode_for_distance(dij, effective_policy)
                mode_matrix[i][j] = mode
                modes_needed.add(mode)

        # Initialize final matrix (diagonal stays 0)
        final_matrix = [[0 for _ in range(n)] for _ in range(n)]

        # Optionally populate API-based durations if enabled; otherwise rely on local estimates only
        dm_time_by_mode = {}
        if USE_DISTANCE_MATRIX:
            # Safe cap for elements per request. Distance Matrix commonly limits to ~100 elements/request.
            max_elements_per_request = 100
            batch_size = min(n, int(max_elements_per_request ** 0.5) or 1)

            # Initialize full matrices for each mode
            dm_time_by_mode = {m: [[None for _ in range(n)] for _ in range(n)] for m in modes_needed}

            # Helper to call Distance Matrix for a given mode and fill dm_time_by_mode
            def cache_key_for_dm(origins, destinations, mode):
                m = hashlib.sha256()
                m.update(('|'.join(origins) + '||' + '|'.join(destinations) + f'||{mode}').encode('utf-8'))
                return m.hexdigest()

            def fill_dm_for_mode(mode):
                for i_start in range(0, n, batch_size):
                    origins = addresses[i_start:i_start + batch_size]
                    for j_start in range(0, n, batch_size):
                        destinations = addresses[j_start:j_start + batch_size]

                        kwargs = { 'origins': origins, 'destinations': destinations, 'mode': mode }
                        if mode == 'transit':
                            kwargs['departure_time'] = 'now'
                            kwargs['transit_mode'] = 'subway'

                        key = cache_key_for_dm(origins, destinations, mode)
                        cache_path = Path(CACHE_DIR) / f"dm_{key}.json"
                        now = time.time()
                        response = None

                        # Cache read
                        try:
                            if cache_path.exists() and (now - cache_path.stat().st_mtime) <= CACHE_TTL_SEC:
                                with open(cache_path, 'r') as f:
                                    response = json.load(f)
                        except Exception:
                            response = None

                        # Live call with retries/backoff if no cache
                        if response is None:
                            attempt = 0
                            while attempt <= API_MAX_RETRIES:
                                try:
                                    response = gmaps.distance_matrix(**kwargs)
                                    # Cache write best-effort
                                    try:
                                        with open(cache_path, 'w') as f:
                                            json.dump(response, f)
                                    except Exception:
                                        pass
                                    break
                                except Exception:
                                    attempt += 1
                                    if attempt > API_MAX_RETRIES:
                                        response = {'rows': []}
                                        break
                                    backoff = (API_BACKOFF_BASE_MS / 1000.0) * (2 ** (attempt - 1)) * (1 + random.random())
                                    time.sleep(backoff)

                        rows = response.get('rows', []) if isinstance(response, dict) else []
                        for oi, row in enumerate(rows):
                            elements = row.get('elements', [])
                            for di_idx, elem in enumerate(elements):
                                gi = i_start + oi
                                gj = j_start + di_idx
                                if gi == gj:
                                    continue
                                try:
                                    status = elem.get('status', 'OK')
                                    if status == 'OK' and 'duration' in elem and 'value' in elem['duration']:
                                        dm_time_by_mode[mode][gi][gj] = int(elem['duration']['value'])
                                except Exception:
                                    pass

            for mode in modes_needed:
                fill_dm_for_mode(mode)

        # Compose final matrix with chosen mode per pair; fallback via haversine speed model
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mode = mode_matrix[i][j] or 'transit'
                dm_val = dm_time_by_mode.get(mode, None)
                travel_seconds = None
                if dm_val is not None:
                    travel_seconds = dm_val[i][j]
                if travel_seconds is None:
                    travel_seconds = estimate_travel_seconds(dist_matrix[i][j], mode)
                final_matrix[i][j] = int(travel_seconds) + int(data['time_on_site'][j])

        return final_matrix
    except Exception as e:
        print(f"Time matrix error: {e}")
        return None


def print_solution(data, manager, routing, solution):
    """Prints solution, now including poster usage."""
    print("Solution:")
    # ... (code for printing dropped tasks is the same)
    
    print("Team itineraries:")
    # Access the time dimension to read cumulative times
    time_dimension = routing.GetDimensionOrDie('time')
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f"Team {vehicle_id + 1} Route:\n"
        route_posters = 0
        
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append((node_index, solution.Value(time_dimension.CumulVar(index))))
            if node_index != data['depot']:
                 route_posters += data['poster_costs'][node_index]
            index = solution.Value(routing.NextVar(index))
        route.append((manager.IndexToNode(index), solution.Value(time_dimension.CumulVar(index))))

        # ... (code for printing route steps is mostly the same)
        for i in range(len(route) - 1):
             start_node, start_time_s = route[i]
             start_time_m = start_time_s // 60
             task_name = data['task_names'][start_node]
             time_on_site_m = data['time_on_site'][start_node] // 60
             poster_cost = data['poster_costs'][start_node]
             plan_output += f"  ({start_time_m//60:02d}h{start_time_m%60:02d}m) Arrive at {task_name}\n"
             if time_on_site_m > 0:
                 plan_output += f"           -> Spend {time_on_site_m} min on site (using {poster_cost} posters).\n"
        # ... (rest of the printing logic)

        total_route_time_h = solution.Value(time_dimension.CumulVar(routing.End(vehicle_id))) / 3600
        plan_output += f"\n  Total Time for Team {vehicle_id + 1}: {total_route_time_h:.2f} hours"
        plan_output += f"\n  Posters Used by Team {vehicle_id + 1}: {route_posters} / {data['poster_capacities'][vehicle_id]}\n"
        print(plan_output)

    # Also export itineraries as CSV for easier viewing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = Path(__file__).parent / 'outputs' / f'run_{timestamp}'
    base_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_output_dir / 'vrp_itinerary.csv'
    try:
        export_itineraries_to_csv(str(csv_path), data, manager, routing, solution)
        print(f"CSV: {csv_path}")
    except Exception as e:
        print(f"CSV export failed: {e}")

    # Write metadata.json with configuration and detected policy
    try:
        # Re-detect policy the same way build_time_matrix does (lightweight duplicate)
        lats = data.get('latitudes') or []
        lngs = data.get('longitudes') or []
        n = len(lats)
        def detect_effective_policy():
            if TRAVEL_POLICY != 'auto':
                return TRAVEL_POLICY
            distances = []
            for i in range(1, n):
                for j in range(i + 1, n):
                    d = haversine_meters(lats[i], lngs[i], lats[j], lngs[j])
                    if d is not None and d > 0:
                        distances.append(d)
            if not distances:
                return 'auto_city'
            distances.sort()
            median = distances[len(distances)//2]
            return 'auto_campus' if median <= WALK_MAX_METERS_CAMPUS else 'auto_city'

        effective_policy = detect_effective_policy()
        metadata = {
            'timestamp': timestamp,
            'travel_policy_requested': TRAVEL_POLICY,
            'travel_policy_effective': effective_policy,
            'walk_max_meters_city': WALK_MAX_METERS_CITY,
            'walk_max_meters_campus': WALK_MAX_METERS_CAMPUS,
            'walk_speed_mps': WALK_SPEED_MPS,
            'transit_speed_mps': TRANSIT_SPEED_MPS,
            'drive_speed_mps': DRIVE_SPEED_MPS,
            'transit_overhead_seconds': TRANSIT_OVERHEAD_SECONDS,
            'drive_overhead_seconds': DRIVE_OVERHEAD_SECONDS,
            'popularity_max': POPULARITY_MAX,
            'poster_scale_base': POSTER_SCALE_BASE,
            'poster_scale_pop_coeff': POSTER_SCALE_POP_COEFF,
            'poster_scale_rating_coeff': POSTER_SCALE_RATING_COEFF,
            'rating_baseline': RATING_BASELINE,
            'rating_range': RATING_RANGE,
            'time_scale_base': TIME_SCALE_BASE,
            'time_pop_coeff': TIME_POP_COEFF,
            'priority_penalty_weight': PRIORITY_PENALTY_WEIGHT,
            'poster_penalty_weight': POSTER_PENALTY_WEIGHT,
            'solver_time_limit_sec': SOLVER_TIME_LIMIT_SEC,
            'solver_random_seed': SOLVER_RANDOM_SEED,
            'first_solution_strategy': FIRST_SOLUTION_STRATEGY,
            'local_search_meta': LOCAL_SEARCH_META,
            'cache_dir': CACHE_DIR,
            'cache_ttl_sec': CACHE_TTL_SEC,
            'api_max_retries': API_MAX_RETRIES,
            'api_backoff_base_ms': API_BACKOFF_BASE_MS
        }
        with open(base_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata: {base_output_dir / 'metadata.json'}")
    except Exception as e:
        print(f"Metadata write failed: {e}")


def export_itineraries_to_csv(csv_path, data, manager, routing, solution):
    """Export per-vehicle itineraries to a CSV file.

    Columns: team, timestamp (HH:MM), location (task name), address, posters_to_expend
    """
    time_dimension = routing.GetDimensionOrDie('time')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["team", "timestamp", "location", "address", "posters_to_expend"])

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                cumul_s = solution.Value(time_dimension.CumulVar(index))
                hours = int(cumul_s // 3600)
                minutes = int((cumul_s % 3600) // 60)
                timestamp = f"{hours:02d}:{minutes:02d}"
                task_name = data['task_names'][node_index]
                address = data['locations'][node_index]
                posters = 0 if node_index == data['depot'] else data['poster_costs'][node_index]
                writer.writerow([vehicle_id + 1, timestamp, task_name, address, posters])
                index = solution.Value(routing.NextVar(index))

            # Add final depot/end entry
            end_index = routing.End(vehicle_id)
            cumul_s = solution.Value(time_dimension.CumulVar(end_index))
            hours = int(cumul_s // 3600)
            minutes = int((cumul_s % 3600) // 60)
            timestamp = f"{hours:02d}:{minutes:02d}"
            end_node = manager.IndexToNode(end_index)
            task_name = data['task_names'][end_node] if end_node < len(data['task_names']) else 'DEPOT'
            address = data['locations'][end_node] if end_node < len(data['locations']) else data['locations'][data['depot']]
            writer.writerow([vehicle_id + 1, timestamp, task_name, address, 0])


def solve_vrp_with_capacity(tasks, constraints):
    """The main VRP solver, now with poster capacity."""
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    data = create_data_model(tasks, constraints)
    
    print("Fetching travel times...")
    time_matrix = build_time_matrix(gmaps, data)
    if time_matrix is None: return

    print("Building model...")
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # --- Time Dimension ---
    def time_callback(from_index, to_index):
        return time_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
    routing.AddDimension(time_callback_index, 0, data['workday_seconds'], True, 'time')
    time_dimension = routing.GetDimensionOrDie('time')

    # --- Poster Dimension ---
    def poster_callback(from_index):
        return data['poster_costs'][manager.IndexToNode(from_index)]
    poster_callback_index = routing.RegisterUnaryTransitCallback(poster_callback)
    routing.AddDimensionWithVehicleCapacity(
        poster_callback_index, 0, data['poster_capacities'], True, 'posters'
    )
    posters_dimension = routing.GetDimensionOrDie('posters')

    # Encourage using available posters (soft lower bound per vehicle based on supply)
    total_posters_available = sum(data['poster_costs'][1:])  # exclude depot
    per_vehicle_target = 0 if data['num_vehicles'] == 0 else (total_posters_available + data['num_vehicles'] - 1) // data['num_vehicles']
    for vehicle_id in range(data['num_vehicles']):
        target = min(data['poster_capacities'][vehicle_id], per_vehicle_target)
        if target > 0:
            posters_dimension.SetCumulVarSoftLowerBound(
                routing.End(vehicle_id), target, 100000
            )


    # --- Penalties for Dropping Tasks ---
    # Heavier penalties for dropping high-priority and high-poster-cost tasks
    workday_hours = max(1, data['workday_seconds'] // 3600)
    for node_index in range(1, len(data['locations'])):
        priority = data['priorities'][node_index]
        poster_cost = data['poster_costs'][node_index]
        # Env-tunable penalty weights
        penalty = int((priority * PRIORITY_PENALTY_WEIGHT + poster_cost * POSTER_PENALTY_WEIGHT) * workday_hours)
        routing.AddDisjunction([manager.NodeToIndex(node_index)], penalty)

    # --- Solver Settings ---
    print("Solving...")
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # First solution
    try:
        search_parameters.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, FIRST_SOLUTION_STRATEGY
        )
    except Exception:
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    # Metaheuristic
    try:
        search_parameters.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, LOCAL_SEARCH_META
        )
    except Exception:
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # Time limit
    search_parameters.time_limit.FromSeconds(int(max(1, SOLVER_TIME_LIMIT_SEC)))
    # Optional seed
    if SOLVER_RANDOM_SEED > 0:
        search_parameters.random_seed = SOLVER_RANDOM_SEED

    # Help the solver finalize shorter routes per vehicle
    # Encourage each vehicle to perform at least some work via a soft lower bound on time
    min_time_per_vehicle_s = 20 * 60  # 20 minutes
    for vehicle_id in range(data['num_vehicles']):
        time_dimension.SetCumulVarSoftLowerBound(
            routing.End(vehicle_id), min_time_per_vehicle_s, 10
        )
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle_id)))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(vehicle_id)))

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution) # This function will need a small update
    else:
        print("No solution.")

############################################################################
# ----- 5. MAIN EXECUTION -----
############################################################################

def main():
    """Entry point of the script."""
    print("VRP Planner")
    
    # This is where you define your high-level goal.
    user_request = "I have 2 teams, each can work for 8 hours and carries 500 posters. We want to hit high-traffic areas frequented by university students and establish a strong presence on the main university campuses in Manhattan."
    
    print(f"Goal: {user_request}")
    
    # Phase 1: Generate the strategic plan
    strategic_plan = generate_strategic_plan(user_request)
    if not strategic_plan: return
        
    # Phase 2 & 3: Discover locations and generate tasks based on the plan
    if not GOOGLE_MAPS_API_KEY:
        print("Missing GOOGLE_MAPS_API_KEY.")
        return
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    tasks = discover_and_generate_tasks(gmaps_client, strategic_plan)
    if not tasks: return
        
    # Phase 4: Solve the VRP with time and poster constraints
    solve_vrp_with_capacity(tasks, strategic_plan['constraints'])

if __name__ == '__main__':
    main()

