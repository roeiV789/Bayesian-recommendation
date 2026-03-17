#flight features:
# 1.price
# 2.departure_time
# 3.duration
# 4.stops
import numpy as np
class BayesianAssistant:
    def __init__(self):
        #we keep a belief state for each feature as a probability distribution over possible values
        self.features = ['price', 'time_penalty', 'duration', 'stops']
        #departure_time impact will be represented by the the time_penalty feature, which will be calculated in get_time_penalty
        #we initialize uniform distributions for each feature
        self.mu = np.array([0.25, 0.25, 0.25, 0.25])  # mean of the distribution
        self.cov = np.eye(4) *0.1 #we assign variances to each feature, assuming that there is an independence between preferences
        self.ideal_time = 9*60 #we assume that the ideal departure time is 9:00 AM(9*60 minutes)

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
            return np.array(raw_flight_data)
        
        def compute_utility_function(self, normalized_flights):
            
            return -np.dot(normalized_flights, self.mu)
        
        def predict_choice_probs(self, normalized_flights):
            
        

