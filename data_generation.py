import json
import random
import numpy as np      

def generate_random_flight_batch(n=4):
    #generate n flights with random features. simulates the generation of one of the synthetic flight sets the user will be choosing from.
    flights = []
    #as close as we can to flight data from tel aviv
    mu = 5.7
    sigma = 0.6
    for _ in range(n):
        #if we look at real flight data, we see that it follows a bell curve with most flights being around the mean and few flights that are very cheap or very expensive.
        #we can use a log-normal distribution to simulate this
        #we need to make sure that we clip the values to a reasonable range, as log-normal can produce very large values
        raw_price = np.random.lognormal(mean = mu, sigma = sigma)
        price = np.clip(raw_price, 200, 1200)  # Clip to a reasonable price range
        #time is an integer, we use randint
        time = random.randint(300, 1300) # 5:00 AM to 9:00 PM
        #we model duration using a normal distribution, and clip to a reasonable range
        raw_duration = int(np.random.normal(loc=270, scale=120)) 
        duration = np.clip(raw_duration, 90, 720) # 1.5 hour to 12 hours
        #it is quite uncommon to have more than 2 stops
        stops = random.choice([0, 1, 2], p = [0.4, 0.5, 0.1])
        flights.append([price, time, duration, stops])
    return flights

