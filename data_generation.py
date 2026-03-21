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


import numpy as np

def generate_reasoning(raw_flight_data, processed_data, choice_idx, prior_weights, posterior_weights, features, ideal_time_mins=9*60):
    """
    Generates a synthetic Chain-of-Thought reasoning string focusing on trade-offs.
    Calls the external `explain_time_penalty` helper to translate cyclical time logic.
    """
    chosen_flight_raw = raw_flight_data[choice_idx]
    chosen_flight_norm = processed_data[choice_idx]
    unselected_indices = [i for i in range(len(raw_flight_data)) if i != choice_idx]
    
    reasoning_parts = []
    
    # 1. The Observation
    reasoning_parts.append(f"The user selected Flight {choice_idx}.")
    
    # 2. Contrastive Trade-offs (Handling standard features first)
    advantages = []
    sacrifices = []
    
    # Locate the index for time_penalty so we can skip it in the standard loop
    time_idx = features.index('time_penalty') if 'time_penalty' in features else 1
    
    for feature_idx, feature_name in enumerate(features):
        if feature_idx == time_idx:
            continue # Skip time penalty here; we handle it below via the helper
            
        chosen_val = chosen_flight_raw[feature_idx]
        rejected_vals = [raw_flight_data[i][feature_idx] for i in unselected_indices]
        
        if chosen_val <= min(rejected_vals):
            advantages.append(feature_name)
        elif chosen_val > min(rejected_vals):
            sacrifices.append(feature_name)

    # Format advantages grammatically
    if advantages:
        if len(advantages) > 2:
            adv_str = ", ".join(advantages[:-1]) + f", and {advantages[-1]}"
        else:
            adv_str = " and ".join(advantages)
        reasoning_parts.append(f"This flight offered the most competitive {adv_str} compared to the alternatives.")
        
    # Format sacrifices grammatically
    if sacrifices:
        if len(sacrifices) > 2:
            sac_str = ", ".join(sacrifices[:-1]) + f", and {sacrifices[-1]}"
        else:
            sac_str = " and ".join(sacrifices)
        reasoning_parts.append(f"However, the user accepted worse options for {sac_str} than were available on other flights.")

    # 3. Call the external helper function for the time penalty
    dep_time_mins = chosen_flight_raw[time_idx]
    time_penalty_val = chosen_flight_norm[time_idx]
    
    time_explanation = explain_time_penalty(dep_time_mins, ideal_time_mins, time_penalty_val)
    reasoning_parts.append(f"Regarding schedule: {time_explanation}")
    
    # 4. Logical Inference
    if advantages and sacrifices:
        reasoning_parts.append(
            f"Choosing this flight despite the trade-offs indicates that the user strongly prioritizes {adv_str} over {sac_str}."
        )
    
    # 5. The Bayesian Update
    weight_diff = posterior_weights - prior_weights
    significant_weight_changes = np.where(np.abs(weight_diff) > 0.05)[0]
    
    if len(significant_weight_changes) > 0:
        effects = []
        for i in significant_weight_changes:
            direction = "increased priority (lower weight)" if weight_diff[i] < 0 else "decreased priority (higher weight)"
            effects.append(f"{features[i]} showed {direction}")
            
        reasoning_parts.append(
            "Updating our belief state to reflect this observation, the expected weights shift significantly: " + ", ".join(effects) + "."
        )
    else:
        reasoning_parts.append(
            "Because this choice aligns with existing uncertainties or was an objectively dominant flight, the expected feature weights remain largely unchanged."
        )

    return " ".join(reasoning_parts)


def explain_time_penalty(departure_time_mins, ideal_time_mins, time_penalty):
    """
    Translates the continuous cyclical time penalty into a semantic 
    explanation for the LLM's Chain-of-Thought.
    """
    # 1. Format the times for readability (e.g., 540 -> "09:00")
    dep_h, dep_m = divmod(departure_time_mins, 60)
    ideal_h, ideal_m = divmod(ideal_time_mins, 60)
    
    dep_str = f"{int(dep_h):02d}:{int(dep_m):02d}"
    ideal_str = f"{int(ideal_h):02d}:{int(ideal_m):02d}"
    
    # 2. Calculate the straightforward "hours away" (accounting for midnight wrap-around)
    raw_diff = abs(departure_time_mins - ideal_time_mins)
    shortest_diff_mins = min(raw_diff, (24 * 60) - raw_diff)
    hours_away = round(shortest_diff_mins / 60.0, 1)
    
    # 3. Categorize the mathematical penalty based on the cyclical distance
    # The penalty is sin(delta/2). 
    # < 0.15 is roughly within 1 hour. < 0.35 is roughly within 3 hours.
    if time_penalty < 0.15:
        severity = "a minimal"
    elif time_penalty < 0.35:
        severity = "a moderate"
    elif time_penalty < 0.70:
        severity = "a significant"
    else:
        severity = "a severe"
        
    # 4. Construct the semantic explanation
    if hours_away == 0:
        return f"Flight departs exactly at the ideal {ideal_str}, incurring no time penalty (0.00)."
    else:
        return f"Flight departs at {dep_str}. Being {hours_away} hours away from the ideal {ideal_str}, it incurs {severity} time penalty ({time_penalty:.2f})."
