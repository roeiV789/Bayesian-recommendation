import json
import random
import numpy as np      

def generate_random_flight_batch(n=4):
    """
    Generates a batch of random flight data.
    parameters:
    n: the number of flights to generate, default is 4
    returns:
    flights: a list of n flights, where each flight is represented as a list of features [price, departure_time, duration, stops]

    
    """
    #generate n flights with random features. simulates the generation of one of the synthetic flight sets the user will be choosing from.
    flights = []
    #as close as we can to flight data from tel aviv
    mu = 5.9
    sigma = 0.4
    while len(flights) < n:
        #if we look at real flight data, we see that it follows a bell curve with most flights being around the mean and few flights that are very cheap or very expensive.
        #we can use a log-normal distribution to simulate this
        #in order to not draw small or negative values, we will draw a new number when receiving a value under 200 or greater than 1200.
        raw_price = np.random.lognormal(mean = mu, sigma = sigma)
        if not (raw_price>=200 and raw_price<1200):
                continue #throw the value away and draw a new one
        price = raw_price
        #time is an integer, we use randint
        time = random.randint(300, 1300) # 5:00 AM to 9:00 PM
        #we model duration using a normal distribution, and throw away values outside of the range
        raw_duration = int(np.random.normal(loc=270, scale=45)) 
        if not (raw_duration>=90 and raw_duration<=720):
            continue #throw the value away and draw a new one
        duration = raw_duration
        #it is quite uncommon to have more than 2 stops
        stops = np.random.choice([0, 1, 2], p = [0.4, 0.5, 0.1])
        flights.append([price, time, duration, stops])
    return flights

def generate_reasoning(processed_data, choice_idx, prior_weights, posterior_weights, prior_belief, posterior_belief, features, user_profiles, belief_noise_threshold=0.005, weight_noise_threshold=0.05):
    """
    Generates a synthetic Chain-of-Thought reasoning string. 
    parameters: 
    processed_data: the normalized flight data that the user chose from, shape is [4,4] (4 flights, 4 features) 
    choice_idx: the index of the flight that the user chose, an integer from 0 
    prior_weights: the expected weights for each feature before the user made their choice, shape is [4] (4 features) 
    posterior_weights: the expected weights for each feature after the user made their choice, shape is [4] (4 features) 
    features: the list of feature names, in the same order as the weights and the processed data, shape is [4] (4 features) 
    user_profiles: the matrix of user profiles, shape is [625, 4] (625 profiles, 4 features) 
    noise_threshold: the threshold for determining significant weight changes, default is 0.05
    returns: 
    reasoning: a string that explains the user's choice from a bayesian perspective
    """

    reasoning_parts = []

    #what was the observation? what was the user's choice?
    reasoning_parts.append(f"The user selected Flight {choice_idx} from the available options.")

    #how can we interpret this observation in terms of the user's preferences?
    reasoning_parts.append(
        "Each possible preference profile assigns a probability to this choice based on how well the flight's features align with its preferences."
    )

    #we calculate the belief shift - the change in probability for each user profile before and after observing the choice.
    #positive value - the choice made the profile more likely, negative value - the choice made the profile less likely
    #the belief starts with a uniform distribution
    belief_shift = posterior_belief - prior_belief
    weight_diff = posterior_weights - prior_weights #how did the weights for the features change as a result of the user's choice?
    # the update can be noisy, so we focus on significant shifts. we can tune the noise threshold.
    significant_indices = np.where(np.abs(belief_shift) > belief_noise_threshold)[0]
    significant_weight_changes = np.where(np.abs(weight_diff) > weight_noise_threshold)[0] #argue only based on significant changes

    if len(significant_indices) > 0 or len(significant_weight_changes) > 0:
        reasoning_parts.append(
            "Profiles that assigned higher likelihood to the chosen flight increased in probability, while others decreased."
        )

        #what is the direction of the shift? we average the shifts of the significant profiles, by using the weights of the profiles that were the most changed by this choice
        if len(significant_indices) > 0:
            avg_shift = np.average(user_profiles[significant_indices], axis=0, weights=np.abs(belief_shift[significant_indices]))
            feature_effects = [] #we use the values of avg_shift to determine how the users preference changed for each feature.
            for i, feature in enumerate(features):
                if abs(avg_shift[i]) > 0.1: #for significant changes
                    direction = "prefer lower" if avg_shift[i] < 0 else "prefer higher" #lower values are represented by negative weights, higher values by positive weights, so the sign determines the preference
                    feature_effects.append(f"{feature} ({direction} values)") #for example: "price (prefer lower values)"

            if feature_effects: 
                #we convert the list into a string that explains the change in preferences for the features, and we add it to the reasoning
                reasoning_parts.append(
                    "As a result, the belief shifts toward profiles that " +
                    ", ".join(feature_effects) + "."
                )
       
        if len(significant_weight_changes) > 0:
            effects = []
            for i in significant_weight_changes:
                direction = "increased" if weight_diff[i] > 0 else "decreased"
                effects.append(f"{features[i]} weight {direction}") #for example: "price weight increased"
            reasoning_parts.append(
                "This is reflected in the expected weights, where " + ", ".join(effects) + "."
            )
    #we must include the else logic in order to teach the model to recognize the case where there are no significant changes
    else:
        #we must provide an explanation also for the case where there are no significant shifts in the belief, in order to teach the model to recognize this case.
        reasoning_parts.append(
            "Most preference profiles already assigned similar probabilities to this choice, so the observation does not strongly distinguish between them."
        )
        reasoning_parts.append(
            "As a result, the posterior remains similar to the prior, reinforcing existing uncertainty about the user's preferences."
        )
        reasoning_parts.append(
            "Accordingly, the expected feature weights remain largely unchanged."
        )

    return " ".join(reasoning_parts)



