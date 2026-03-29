import json
import random
import numpy as np
from tqdm import tqdm #we use tqdm to show a progress bar during dataset generation
from symbolic_bayesian_model import BayesianAssistant
from SimulatedTraveler import SimulatedTraveler
from data_generation import generate_random_flight_batch, generate_hypothesis_reasoning


def format_hypothesis_prompt(flights, chosen_idx, new_flights, ideal_time_mins=9*60):
    """
    Formats the context strictly into the target inference and ranking prompt.
    """
    ideal_str = f"{ideal_time_mins//60:02d}:{ideal_time_mins%60:02d}"
    
    prompt = "A user was presented with the following flights:\n\n"
    for i, f in enumerate(flights):
        prompt += f"Flight {i}: Price={f[0]}, Departure={f[1]//60:02d}:{f[1]%60:02d}, Duration={f[2]}, Stops={f[3]}\n"
        
    prompt += f"\nThe user selected: Flight {chosen_idx}\n\n"
    
    prompt += "Step 1: Consider possible explanations for the user's choice:\n"
    prompt += "- Prefers cheaper flights\n"
    prompt += "- Prefers shorter duration\n"
    prompt += "- Prefers fewer stops\n"
    prompt += f"- Prefers departure close to {ideal_str}\n\n"
    
    prompt += "Step 2: Evaluate which explanations are supported or contradicted by the choice.\n\n"
    prompt += "Step 3: Conclude what the user likely values.\n\n"
    prompt += "Now, based on this inferred preference, rank the following new flights:\n\n"
    
    labels = ['A', 'B', 'C', 'D']
    for label, f in zip(labels, new_flights):
        prompt += f"Flight {label}: Price={f[0]}, Departure={f[1]//60:02d}:{f[1]%60:02d}, Duration={f[2]}, Stops={f[3]}\n"
        
    prompt += "\nProvide your reasoning and final ranking."
    return prompt

def generate_synthetic_dataset(num_users=10000, interactions_per_user=4, output_file="bayesian_flight_data.jsonl"):
    assistant = BayesianAssistant()
    dataset = []
    
    # New system prompt focusing on Bayesian hypothesis testing, not numeric weights
    system_message = "You are a travel assistant that learns user preferences from past choices and uses this to recommend future flights to the same user. Reason about the user's preferences by comparing possible explanations for the user's behavior and updating your beliefs accordingly."
    
    print(f"Simulating {num_users} users with {interactions_per_user} interactions each...")
    
    for user_id in tqdm(range(num_users)):
        true_weights = random.choice(assistant.user_profiles)
        traveler = SimulatedTraveler(true_weights, assistant, noise_temp=0.15)
        assistant.reset_belief_state()
        
        for interaction_step in range(interactions_per_user):
            # 1. Generate Context Flights (The Observation)
            raw_flights = generate_random_flight_batch(n=4)
            
            # Calculate prior probabilities to check for "No-Update/Flat Likelihood" state
            prior_likelihoods = assistant.predict_choice_probs(raw_flights)
            prior_expected_probs = np.sum(prior_likelihoods * assistant.belief_state[:, None], axis=0)
            
            # 2. Simulated user makes a choice
            choice_idx, _, _ = traveler.evaluate_and_choose(raw_flights)
            
            # 3. Update internal belief model
            assistant.update_belief_state(raw_flights, choice_idx)
            
            # 4. Generate Target Flights (The Downstream Task)
            new_flights = generate_random_flight_batch(n=4)
            
            # Predict how the user would rank these new flights using the UPDATED posterior
            new_likelihoods = assistant.predict_choice_probs(new_flights)
            new_expected_probs = np.sum(new_likelihoods * assistant.belief_state[:, None], axis=0)
            
            # 5. Build prompt and reasoning
            user_prompt = format_hypothesis_prompt(raw_flights, choice_idx, new_flights, assistant.ideal_time)
            
            llm_completion = generate_hypothesis_reasoning(
                flights=raw_flights,
                chosen_index=choice_idx,
                prior_probs=prior_expected_probs,
                new_flights=new_flights,
                new_flight_expected_probs=new_expected_probs,
                ideal_time_mins=assistant.ideal_time
            )
            
            dataset.append({
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": llm_completion}
                ]
            })

    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\nSuccessfully generated {len(dataset)} training examples and saved to {output_file}.")
    return dataset


# Run it
if __name__ == "__main__":
    generate_synthetic_dataset(num_users=10000, interactions_per_user=4)