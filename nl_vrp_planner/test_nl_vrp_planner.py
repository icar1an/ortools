import os
import unittest
from unittest import mock

# Import functions from nl_vrp_planner
from nl_vrp_planner import (
    haversine_meters,
    estimate_travel_seconds,
    choose_mode_for_distance,
    compute_dynamic_poster_and_time,
    create_data_model,
    build_time_matrix,
    solve_vrp_with_capacity,
    TRAVEL_POLICY,
    WALK_MAX_METERS_CITY,
    WALK_MAX_METERS_CAMPUS,
)

class TestVRPPlannerHelpers(unittest.TestCase):
    def test_haversine_zero(self):
        self.assertEqual(haversine_meters(40.0, -73.0, 40.0, -73.0), 0.0)

    def test_haversine_known_distance(self):
        # Roughly 1 km apart in Manhattan
        d = haversine_meters(40.741, -73.989, 40.749, -73.987)
        self.assertTrue(800 <= d <= 1400)

    def test_choose_mode_auto_city(self):
        self.assertEqual(choose_mode_for_distance(WALK_MAX_METERS_CITY - 1, 'auto_city'), 'walking')
        self.assertEqual(choose_mode_for_distance(WALK_MAX_METERS_CITY + 1, 'auto_city'), 'transit')

    def test_choose_mode_auto_campus(self):
        self.assertEqual(choose_mode_for_distance(WALK_MAX_METERS_CAMPUS - 1, 'auto_campus'), 'walking')
        self.assertEqual(choose_mode_for_distance(WALK_MAX_METERS_CAMPUS + 1, 'auto_campus'), 'driving')

    def test_estimate_travel_seconds_walking(self):
        secs = estimate_travel_seconds(1400, 'walking')  # ~ 1 km -> ~ 1000 s at 1 m/s
        self.assertTrue(600 <= secs <= 1800)

    def test_estimate_travel_seconds_transit_and_driving(self):
        s_transit = estimate_travel_seconds(5000, 'transit')
        s_driving = estimate_travel_seconds(5000, 'driving')
        self.assertTrue(s_transit > 0 and s_driving > 0)
        # Driving should typically be faster than transit at 5 km with defaults
        self.assertLess(s_driving, s_transit + 600)  # allow overlap due to overheads

    def test_compute_dynamic_poster_and_time_baseline(self):
        posters, time_m = compute_dynamic_poster_and_time(5, 5, rating=3.0, user_count=0)
        self.assertTrue(posters >= 1)
        self.assertTrue(time_m >= 0)

    def test_compute_dynamic_poster_and_time_popular_high_rating(self):
        posters_pop, time_pop = compute_dynamic_poster_and_time(5, 5, rating=4.8, user_count=800)
        posters_base, time_base = compute_dynamic_poster_and_time(5, 5, rating=3.0, user_count=0)
        self.assertGreaterEqual(posters_pop, posters_base)
        self.assertGreaterEqual(time_pop, time_base)


class TestVRPPlannerIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure outputs directory exists
        self.base_dir = os.path.dirname(__file__)
        self.outputs_dir = os.path.join(self.base_dir, 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)

    def _fake_distance_matrix(self, origins, destinations, mode=None, **kwargs):
        # Force fallback by returning NOT_FOUND statuses
        rows = []
        for _ in origins:
            row = {'elements': [{'status': 'NOT_FOUND'} for _ in destinations]}
            rows.append(row)
        return {'rows': rows}

    def test_build_time_matrix_fallback_walking(self):
        # Patch travel policy to walking_only for deterministic mode
        import nl_vrp_planner as mod
        old_policy = mod.TRAVEL_POLICY
        mod.TRAVEL_POLICY = 'walking_only'
        try:
            data = {
                'locations': ['A', 'B', 'C'],
                'time_on_site': [0, 60, 120],  # seconds
                'latitudes': [40.7484, 40.7480, 40.7490],
                'longitudes': [-73.9857, -73.9870, -73.9865],
            }
            with mock.patch('nl_vrp_planner.googlemaps.Client') as MockClient:
                client = MockClient.return_value
                client.distance_matrix.side_effect = self._fake_distance_matrix
                matrix = build_time_matrix(client, data)
            self.assertEqual(len(matrix), 3)
            self.assertEqual(len(matrix[0]), 3)
            # Diagonal should be 0
            self.assertEqual(matrix[0][0], 0)
            # Off-diagonals include service time at destination
            self.assertGreaterEqual(matrix[0][1], data['time_on_site'][1])
            self.assertGreaterEqual(matrix[1][2], data['time_on_site'][2])
        finally:
            mod.TRAVEL_POLICY = old_policy

    def test_end_to_end_solve_and_outputs(self):
        # Minimal deterministic scenario with 1 team and 2 tasks
        constraints = {
            'num_teams': 1,
            'workday_hours': 1,
            'posters_per_team': 50,
        }
        tasks = [
            {
                'name': 'DEPOT', 'address': 'Depot Address', 'priority': 100,
                'time_on_site_min': 0, 'poster_cost': 0,
                'lat': 40.7484, 'lng': -73.9857
            },
            {
                'name': 'Spot A', 'address': 'A Address', 'priority': 50,
                'time_on_site_min': 2, 'poster_cost': 2,
                'lat': 40.7480, 'lng': -73.9870
            },
            {
                'name': 'Spot B', 'address': 'B Address', 'priority': 50,
                'time_on_site_min': 1, 'poster_cost': 1,
                'lat': 40.7490, 'lng': -73.9865
            },
        ]

        before_runs = set(os.listdir(self.outputs_dir))
        with mock.patch('nl_vrp_planner.googlemaps.Client') as MockClient:
            client = MockClient.return_value
            client.distance_matrix.side_effect = self._fake_distance_matrix
            solve_vrp_with_capacity(tasks, constraints)

        after_runs = set(os.listdir(self.outputs_dir))
        new_dirs = list(after_runs - before_runs)
        self.assertTrue(len(new_dirs) >= 1)
        # Pick the most recent directory
        run_dirs = [os.path.join(self.outputs_dir, d) for d in new_dirs]
        latest = max(run_dirs, key=lambda p: os.path.getmtime(p))
        csv_path = os.path.join(latest, 'vrp_itinerary.csv')
        meta_path = os.path.join(latest, 'metadata.json')
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(meta_path))
        # CSV should have header and at least 3 lines (depot, a stop, end)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        self.assertGreaterEqual(len(lines), 3)

if __name__ == '__main__':
    unittest.main()
