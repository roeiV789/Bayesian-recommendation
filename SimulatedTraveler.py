import numpy as np
from symbolic_bayesian_model import BayesianAssistant

class SimulatedTraveler:
    def __init__(self, target_weights, assistant_ref, noise_temp=0.15):
        """
        target_weights: numpy array of shape (4,) representing ground-truth preferences.
                        e.g., [-1.0, 0.5, -0.5, 0.0]
        assistant_ref: reference to the BayesianAssistant to ensure feature transformations
                       (like time_penalty and normalization min/max) are identical.
        noise_temp: controls how 'rational' the user is.
        """
        self.target_weights = np.array(target_weights, dtype=float)
        self.assistant = assistant_ref
        self.noise_temp = noise_temp

    def evaluate_and_choose(self, raw_flight_data):
        """
        Calculates utility for each flight and makes a probabilistic choice.
        Returns the index of the chosen flight and the normalized features used.
        """
        # 1. Transform and normalize features using the exact same logic as the assistant
        normalized_features = self.assistant.preprocess_flights(raw_flight_data)
        
        # 2. Calculate ground-truth utility: U = w * x
        utilities = np.dot(normalized_features, self.target_weights)
        
        # 3. Apply temperature-scaled softmax for probabilistic choice
        scaled_utilities = utilities / self.noise_temp
        
        # Shift for numerical stability
        shifted_utilities = scaled_utilities - np.max(scaled_utilities)
        exp_utilities = np.exp(shifted_utilities)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # 4. Sample a choice based on the calculated probabilities
        choice_idx = np.random.choice(len(raw_flight_data), p=probabilities)
        
        return choice_idx, normalized_features, probabilities