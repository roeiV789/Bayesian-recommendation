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
        price = round(raw_price,2)
        #time is an integer, we use randint
        time = random.randint(300, 1300) # 5:00 AM to 9:00 PM
        #we model duration using a normal distribution, and throw away values outside of the range
        raw_duration = int(np.random.normal(loc=270, scale=45)) 
        if not (raw_duration>=90 and raw_duration<=720):
            continue #throw the value away and draw a new one
        duration = raw_duration
        #it is quite uncommon to have more than 2 stops
        stops = int(np.random.choice([0, 1, 2], p = [0.4, 0.5, 0.1]))
        flights.append([price, time, duration, stops])
    return flights

def evaluate_hypotheses(flights, chosen_index, is_low_info, ideal_time_mins=9*60):
    """
    Evaluates each hypothesis programmatically. Handles weak/uninformative cases.
    """
    chosen = flights[chosen_index] #selects the flight the user chose.
    evaluations = {} #dictionary to store the evaluation of each hypothesis.

    #build lists for each of the features across all flights
    prices = [f[0] for f in flights]
    durations = [f[2] for f in flights]
    stops = [f[3] for f in flights]
    
    #calculate the distance of each flight's depparture time from the ideal time.
    time_distances = [min(abs(f[1] - ideal_time_mins), (24*60) - abs(f[1] - ideal_time_mins)) for f in flights] 
    
    evaluate_prices(evaluations, prices, chosen, is_low_info, chosen_index)
    evaluate_durations(evaluations, durations, chosen, is_low_info, chosen_index)
    evaluate_stops(evaluations, stops, chosen, is_low_info, chosen_index)
    evaluate_schedule(evaluations, time_distances, chosen_index, is_low_info, ideal_time_mins)

    return evaluations

def evaluate_prices(evaluations, prices, chosen, low_info_flag, chosen_index):
    if low_info_flag and (max(prices) - min(prices)) < 25: #if the price range is less than $25, we consider it a low information scenario for price preference.
        evaluations["Prefers low cost flights"] = ("Not enough information to determine", "price differences are minimal across the options.")
    elif chosen[0] == min(prices):
        evaluations["Prefers low cost flights"] = ("Supported", f"Flight {chosen_index} is the cheapest option.")
    elif chosen[0] == max(prices):
        evaluations["Prefers low cost flights"] = ("Contradicted", f"Flight {chosen_index} is the most expensive option.")
    else:
        evaluations["Prefers low cost flights"] = ("evidence doesn't support this hypothesis", "Lower cost options were available.")

def evaluate_durations(evaluations, durations, chosen, low_info_flag, chosen_index):
    if low_info_flag and (max(durations) - min(durations)) < 40: #if the duration range is less than 40 minutes, we consider it a low information scenario for duration preference.
        evaluations["Prefers shorter duration"] = ("inconclusive", "Differences in duration are negligible.")
    elif chosen[2] == min(durations):
        evaluations["Prefers shorter duration"] = ("Supported", f"Flight {chosen_index} takes the least amount of time.")
    elif chosen[2] == max(durations):
        evaluations["Prefers shorter duration"] = ("Contradicted", f"Flight {chosen_index} is the longest option.")
    else:
        evaluations["Prefers shorter duration"] = ("Not supported", "Shorter flights were available in the option set.")

def evaluate_stops(evaluations, stops, chosen, low_info_flag, chosen_index):
    if low_info_flag and max(stops) == min(stops):
        evaluations["Prefers fewer stops"] = ("Not enough information to determine", "All flights have the same number of stops.")
    elif chosen[3] == min(stops):
        evaluations["Prefers fewer stops"] = ("Supported", f"Flight {chosen_index} has the minimum number of stops among the options.")
    else:
        evaluations["Prefers fewer stops"] = ("Not supported", "Flights with fewer stops were available.")

def evaluate_schedule(evaluations, time_distances, chosen_index, low_info_flag, ideal_time_mins=9*60):
    ideal_str = f"{ideal_time_mins//60:02d}:{ideal_time_mins%60:02d}"
    if low_info_flag and (max(time_distances) - min(time_distances)) < 60: #if the time penalty range is less than 1 hour, we consider it a low information scenario for schedule preference.
        evaluations[f"Prefers departure close to {ideal_str}"] = ("inconclusive", "departure times are in the same range")
    elif time_distances[chosen_index] == min(time_distances):
        evaluations[f"Prefers departure close to {ideal_str}"] = ("Supported", f"Flight {chosen_index} departs closest to the ideal time of {ideal_str}.")
    else:
        evaluations[f"Prefers departure close to {ideal_str}"] = ("Not supported", f"Other options departed closer to the ideal time of {ideal_str}.")

def generate_hypothesis_reasoning(flights, chosen_index, prior_probs, new_flights, new_flight_expected_probs, ideal_time_mins=9*60):
    """
    Generates the target chain-of-thought based on Bayesian hypothesis testing 
    and outputs a final ranking for a downstream task.
    """
    # Detect "No-Update" case: Likelihood flatness method (max-min < epsilon)
    is_low_info = (np.max(prior_probs) - np.min(prior_probs)) < 0.15 #prior_probs is the flight distribution
    
    # Step 2: Generate evaluations
    evaluations = evaluate_hypotheses(flights, chosen_index, is_low_info, ideal_time_mins)
    
    reasoning = "Step 2: Evaluate explanations\n\n"
    for hyp, (status, explanation) in evaluations.items():
        reasoning += f"- {hyp}: {status}. {explanation}\n"
        
    # Step 3: Conclusion
    reasoning += "\nStep 3: Conclusion\n\n"
    
    # extract the three status categories for each hypothesis and format them
    supported = [clean_hyp(k, ideal_time_mins) for k, (status, _) in evaluations.items() if status == "Supported"]
    contradicted = [clean_hyp(k, ideal_time_mins) for k, (status, _) in evaluations.items() if status == "Contradicted"]
    not_supported = [clean_hyp(k, ideal_time_mins) for k, (status, _) in evaluations.items() if status == "Not supported"]
    inconclusive = [clean_hyp(k, ideal_time_mins) for k, (status, _) in evaluations.items() if status in ["inconclusive", "Not enough information to determine"]]

    # Scenario A: ALL hypotheses are inconclusive (Total Low Information)
    if len(inconclusive) == len(evaluations):
        reasoning += f"The available options are similar or the trade-offs are balanced, providing inconclusive evidence regarding {', '.join(inconclusive)}. Therefore, no significant update to the belief about the user's preferences can be made.\n"


    # Scenario B: Mixed Information
    else:
        
        if supported:
            reasoning += f"The user's choice strongly indicates a preference for {', '.join(supported)}"
        else:
            reasoning += "The user's choice reflects a complex trade-off without a single strongly dominant preference"

        other = []
        if not_supported:
            other.append(f"showing no specific priority for {', '.join(not_supported)}")
        if contradicted:
            other.append(f"does not prioritize {', '.join(contradicted)} at all")
        if inconclusive:
            other.append(f"providing inconclusive evidence regarding the user's preference for {', '.join(inconclusive)}")
        
        if other:
            if len(other) == 1:
                reasoning += f", while {other[0]}.\n"
            elif len(other) == 2:
                reasoning += f", while {other[0]} and {other[1]}.\n"
            else:
                reasoning += f", while {', '.join(other[:-1])}, and {other[-1]}.\n"
        else:
            reasoning += ".\n"
        
    # Ranking new flights based on the updated belief state.
    reasoning += "\nRanking new flights:\n\n"
    
    if is_low_info:
        reasoning += "Since preferences are unclear, prioritizing flights with balanced trade-offs based on general population priors.\n\n"
    
    ranking_indices = np.argsort(new_flight_expected_probs)[::-1] #get a list of the indices of the flights sorted by their expected probabilities.
    labels = ['A', 'B', 'C', 'D']
    
    for idx in ranking_indices:
        lbl = labels[idx]
        reasoning += f"- Flight {lbl}: Expected selection probability {new_flight_expected_probs[idx]:.2f}.\n"

    reasoning += "\nFinal Ranking:\n"
    for i, idx in enumerate(ranking_indices):
         reasoning += f"{i+1}. Flight {labels[idx]}\n"

    return reasoning

def clean_hyp(str_to_clean, ideal_time_mins):
            return str_to_clean.replace("Prefers ", "").replace(f" departure close to {ideal_time_mins//60:02d}:{ideal_time_mins%60:02d}", " schedule")