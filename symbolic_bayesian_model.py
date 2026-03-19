#flight features:
# 1.price
# 2.departure_time
# 3.duration
# 4.stops
import numpy as np
import itertools
class BayesianAssistant:
    def __init__(self, num_levels = 5):
        #we keep a belief state for each feature as a probability distribution over possible values
        self.features = ['price', 'time_penalty', 'duration', 'stops']
        self.ideal_time = 9*60 #we assume that the ideal departure time is 9:00 AM(9*60 minutes)
        #departure_time impact will be represented by the the time_penalty feature, which will be calculated in get_time_penalty
        
        #we use weights that can take values seperated into levels of "caring"
        #like the google research paper, we will use 5 weight layers:
        # strongly prefer lower values, weakly prefer lower values, neutral, weakly prefer higher values, strongly prefer higher values 
        weight_levels = np.linspace(-1.0, 1.0, num_levels)

        #create the grid of weight combinations for the 4 features, i.e the user profiles
        #shape is [625, 4] (5^4 = 625) where each row is a user profile and each column is the weight for a feature
        grid = list(itertools.product(weight_levels, repeat=len(self.features)))
        self.user_profiles = np.array(grid)
        self.num_profiles = self.user_profiles.shape[0]

        #initialize prior using a uniform distribution over the user profiles
        self.prior = np.ones(self.num_profiles) / self.num_profiles
        self.belief_state = self.prior.copy() #we initiate the belief state each time by creating a copy of the prior, so that we can reset it for new users

    def get_time_penalty(self, flight_time_min):
        #we represent the time as hour*60 + minutes
        #we use cylical encoding for the time_penalty similar to the strategy used in the google paper
        #convert to radians - x=cos, y=sin
        angle = (flight_time_min / (24*60)) * 2 * np.pi
        angle_ideal = (self.ideal_time / (24*60)) * 2 * np.pi
        dist = np.sqrt((np.cos(angle)-np.cos(angle_ideal))**2 + (np.sin(angle)-np.sin(angle_ideal))**2)
        return dist/2.0 #normalize to [0,1]
    
    def normalize_flight_data(self, raw_flight_data):

        #we use min-max scaling of the data relative to the current flights, such that the cheapest flight will have a feature value of 0 and the most expensive flight will have a feature value of 1
        #x_normalized = x - min(x) / (max(x) - min(x))
        data = np.array(raw_flight_data, dtype=float)
        #calculate the minimum and maximum for each of the features
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        col_range = max_values - min_values
        #avoid division by zero by setting the range to 1 for features where all values are the same
        col_range[col_range == 0] = 1.0
        return (data-min_values) / col_range
        
    def compute_utility_function(self, normalized_flights):
            
        #calculate the utility of each flight for each user
        return np.dot(self.user_profiles, normalized_flights.T)
        
    def predict_choice_probs(self, raw_flight_data):
        #for each flight, the probability P(i) is calculated as: P(i) = exp(utility_i) / sum_j exp(utility_j)
        #we use softmax because it converts the utilities into a probability distribution, scales the utilities that highlights the relative differences, and we can calculate it in a numerically stable way
        normalized_flights = self.preprocess_flights(raw_flight_data)
        utilities = self.compute_utility_function(normalized_flights)
        #we calculate the probability using the softmax function
        #we use the shifted softmax trick in order to deal with underflow, as we may have negative utilities with large magnitudes -> underflow to zero
        max_utilities = np.max(utilities, axis=1, keepdims=True)
        exp_utilities = np.exp(utilities - max_utilities)
        #normalize by the sum of exponentials for probabilities
        likelihoods = exp_utilities / np.sum(exp_utilities, axis = 1, keepdims=True)
        return likelihoods
    
    def update_belief_state(self, raw_flight_data, chosen_index):
        #we want to update the belief state which is the posterior distribution
        #using bayes rule:
        #P(user_profiles | choice, flights) = P(choice | flights, user_profiles) * P(user_profiles) / P(choice | flights)
        #first we get the value P(choice | flights, user_profiles) for each user profile
        likelihoods = self.predict_choice_probs(raw_flight_data)
        #index for the chosen flight
        cur_likelihood = likelihoods[:, chosen_index]
        #now we use bayes rule to update the belief state
        
        posterior_unnormalized = cur_likelihood * self.belief_state
        #normalize
        self.belief_state = posterior_unnormalized / np.sum(posterior_unnormalized)
    
    def get_expected_weights(self):
        #here we want to get the weights the user has for each feature, so that the llm can use this to generate a recommendation
        #we calculate the expected weights by taking a weighted average of the user profiles using the belief state as weights
        return np.average(self.user_profiles, axis=0, weights=self.belief_state)
    
    def reset_belief_state(self):
        """Resets the engine for a new simulated traveler."""
        self.belief_state = self.prior.copy()
    def preprocess_flights(self, raw_flight_data):
        """Converts raw departure time into the cyclical time penalty before normalizing."""
        processed_data = []
        for flight in raw_flight_data:
            price, dep_time, duration, stops = flight
            time_pen = self.get_time_penalty(dep_time)
            processed_data.append([price, time_pen, duration, stops])
        
        return self.normalize_flight_data(processed_data)
