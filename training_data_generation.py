import json
import random
import numpy as np
from tqdm import tqdm #we use tqdm to show a progress bar during dataset generation
from symbolic_bayesian_model import BayesianAssistant
from SimulatedTraveler import SimulatedTraveler
from data_generation import generate_random_flight_batch, generate_reasoning


def format_llm_prompt(flights, chosen_idx, prior_weights, features):
    """
    Formats the context into a prompt for the LLM.
    The LLM needs to see the options, the choice, and what it currently 'knows'.
    """
    prompt = """You are a travel assistant that helps a user choose a flight. 
                Please analyze the user's flight selection and use it to update your understanding of the user's preferences.
                Focus on the change in the user's preferences based on the choice they made, and explain the trade-offs that the user made in their choice.\n\n""" 
    prompt += f"Current Expected Feature Weights (Prior):\n"
    for feat, weight in zip(features, prior_weights):
        prompt += f"- {feat}: {weight:.2f}\n"
    
    prompt += "\nAvailable Flights [Price, Dep_Time (mins), Duration (mins), Stops]:\n"
    for i, f in enumerate(flights):
        prompt += f"Flight {i}: {f}\n"
        
    prompt += f"\nUser Selected: Flight {chosen_idx}\n"
    prompt += "\nProvide the reasoning for this choice and the updated expected weights."
    
    return prompt

def generate_synthetic_dataset(num_users=10000, interactions_per_user=4, output_file="bayesian_flight_data.jsonl"):
    """
    Runs the simulation and exports a JSONL dataset for LLM fine-tuning.
    """
    assistant = BayesianAssistant()
    dataset = []
    

    system_message = """Explanation about feature weights: 
                    Weights are continuous values bounded between -1.00 and 1.00. 
                    Negative weights indicate a preference for lower values, the more negative the weight the stronger the preference for lower values 
                    Positive weights indicate a preference for higher values, the more positive the weight the stronger the preference for higher values.
                    A value near zero indicates that the user is neutral regarding that feature or we lack data.
                    The time_penalty feature measures distance from an ideal departure time. A negative weight for time_penalty indicates a preference for flights that are closer to the ideal time."""
    
    print(f"Simulating {num_users} users with {interactions_per_user} interactions each...")
    
    for user_id in tqdm(range(num_users)):
        # 1. Sample a ground-truth profile for this simulated user
        true_weights = random.choice(assistant.user_profiles)
        
        # 2. Initialize the traveler and reset the assistant's memory
        traveler = SimulatedTraveler(true_weights, assistant, noise_temp=0.15)
        assistant.reset_belief_state()
        
        # 3. Simulate a sequence of searches
        for interaction_step in range(interactions_per_user):
            # Generate options
            raw_flights = generate_random_flight_batch(n=4)
            processed_flights = assistant.preprocess_flights(raw_flights)
            
            # Record the prior before the update
            prior_weights = assistant.get_expected_weights()
            
            # Traveler makes a choice
            choice_idx, _, _ = traveler.evaluate_and_choose(raw_flights)
            
            # Assistant updates its beliefs based on the choice
            assistant.update_belief_state(raw_flights, choice_idx)
            posterior_weights = assistant.get_expected_weights()
            
            # 4. Generate the Teacher's Chain-of-Thought
            cot_reasoning = generate_reasoning(
                raw_flight_data=raw_flights,
                processed_data=processed_flights,
                choice_idx=choice_idx,
                prior_weights=prior_weights,
                posterior_weights=posterior_weights,
                features=assistant.features,
                ideal_time_mins=assistant.ideal_time
            )
            
            # 5. Format for LLM training (Instruction / Response format)
            user_prompt = format_llm_prompt(raw_flights, choice_idx, prior_weights, assistant.features)
            
            # The completion is the CoT followed by the explicit final weights
            final_weights_str = "\n\nUpdated Expected Weights (Posterior):\n" + "\n".join(
                [f"- {feat}: {w:.2f}" for feat, w in zip(assistant.features, posterior_weights)]
            )
            llm_completion = cot_reasoning + final_weights_str
            
            # Append to dataset
            dataset.append({
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": llm_completion}
                ]
            })

    # 6. Save to disk
    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\nSuccessfully generated {len(dataset)} training examples and saved to {output_file}.")
    return dataset

# Run it
if __name__ == "__main__":
    generate_synthetic_dataset(num_users=10000, interactions_per_user=4)