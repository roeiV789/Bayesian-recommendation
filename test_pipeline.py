#AI generated unit tests for the bayesian assistant+simulated traveler+data generation system.


import unittest
import numpy as np

# Import from your newly separated modules
from symbolic_bayesian_model import BayesianAssistant
from SimulatedTraveler import SimulatedTraveler
from data_generation import generate_random_flight_batch, generate_reasoning

class TestBayesianSystem(unittest.TestCase):
    def setUp(self):
        # We use a smaller num_levels for faster test execution (3^4 = 81 profiles)
        self.assistant = BayesianAssistant(num_levels=3)
        
        # A traveler who strongly prefers lower prices (-1.0) and fewer stops (-1.0),
        # but is neutral to departure time (0.0) and duration (0.0)
        self.target_profile = [-1.0, 0.0, 0.0, -1.0] 
        self.traveler = SimulatedTraveler(self.target_profile, self.assistant, noise_temp=0.1)

    def test_flight_generation_bounds(self):
        """Ensure the generated flights adhere to expected physical and logical bounds."""
        n_flights = 10
        flights = generate_random_flight_batch(n=n_flights)
        
        self.assertEqual(len(flights), n_flights)
        for flight in flights:
            price, dep_time, duration, stops = flight
            self.assertTrue(200 <= price <= 1200, "Price out of bounds")
            self.assertTrue(300 <= dep_time <= 1300, "Departure time out of bounds")
            self.assertTrue(90 <= duration <= 720, "Duration out of bounds")
            self.assertIn(stops, [0, 1, 2], "Invalid number of stops")

    def test_cyclical_time_penalty(self):
        """Verify the cosine/sine distance logic for departure times."""
        ideal_time = self.assistant.ideal_time # 540 mins (9:00 AM)
        opposite_time = ideal_time + (12 * 60) # 1980 mins (9:00 PM)
        
        penalty_ideal = self.assistant.get_time_penalty(ideal_time)
        penalty_opposite = self.assistant.get_time_penalty(opposite_time)
        
        self.assertEqual(penalty_ideal, 0.0, "Ideal time should have 0 penalty")
        self.assertTrue(penalty_opposite > 0.0, "Opposite time must have a positive penalty")
        self.assertTrue(penalty_opposite <= 1.0, "Penalty must be normalized to [0,1]")

    def test_traveler_choice_distribution(self):
        """Check that the traveler produces a valid probability distribution."""
        flights = generate_random_flight_batch(n=4)
        choice_idx, norm_features, probabilities = self.traveler.evaluate_and_choose(flights)
        
        self.assertTrue(0 <= choice_idx < 4)
        self.assertEqual(norm_features.shape, (4, 4))
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5, msg="Probabilities must sum to 1")
        self.assertTrue(np.all(probabilities >= 0), "Probabilities cannot be negative")

    def test_belief_state_update_direction(self):
        """
        Create a rigged scenario to ensure the assistant's belief updates correctly.
        Flight 0 is incredibly cheap and direct. Flight 1 is expensive with stops.
        If the user picks Flight 0, the expected weight for price should decrease.
        """
        rigged_flights = [
            [200, 540, 120, 0],   # Amazing flight
            [1200, 540, 120, 2],  # Terrible flight
            [1100, 540, 120, 2],  # Terrible flight
            [1150, 540, 120, 2]   # Terrible flight
        ]
        
        prior_weights = self.assistant.get_expected_weights()
        
        self.assistant.update_belief_state(rigged_flights, chosen_index=0)
        posterior_weights = self.assistant.get_expected_weights()
        
        self.assertTrue(posterior_weights[0] < prior_weights[0], "Expected price weight did not shift downward")
        self.assertTrue(posterior_weights[3] < prior_weights[3], "Expected stops weight did not shift downward")

    def test_generate_reasoning_output(self):
        """Ensure the CoT text generation runs without throwing errors."""
        flights = generate_random_flight_batch(n=4)
        prior_weights = self.assistant.get_expected_weights()
        prior_belief = self.assistant.belief_state.copy()
        
        choice_idx, norm_features, _ = self.traveler.evaluate_and_choose(flights)
        self.assistant.update_belief_state(flights, choice_idx)
        
        posterior_weights = self.assistant.get_expected_weights()
        posterior_belief = self.assistant.belief_state.copy()
        
        reasoning = generate_reasoning(
            norm_features, choice_idx, prior_weights, posterior_weights,
            prior_belief, posterior_belief, self.assistant.features, self.assistant.user_profiles
        )
        
        self.assertIsInstance(reasoning, str)
        self.assertTrue(len(reasoning) > 10, "Reasoning string is suspiciously short")

if __name__ == '__main__':
    unittest.main()