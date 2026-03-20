#AI generated script to check the data generation and reasoning generation logic
import numpy as np
from symbolic_bayesian_model import BayesianAssistant
from SimulatedTraveler import SimulatedTraveler
from data_generation import generate_random_flight_batch, generate_reasoning

def run_sequential_diagnostic(num_turns=4):
    assistant = BayesianAssistant(num_levels=5)
    
    # We set a specific profile to see if the reasoning catches it.
    # Profile: Strongly hates high prices (-1.0), Neutral to time/duration (0.0), Strongly hates stops (-1.0)
    target_profile = [-1.0, 0.0, 0.0, -1.0]
    traveler = SimulatedTraveler(target_profile, assistant, noise_temp=0.1)
    
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC: SEQUENTIAL SYNTHETIC DATA GENERATION")
    print("="*60)
    print(f"Ground Truth User Profile: {target_profile} (Price, Time, Duration, Stops)\n")

    for turn in range(num_turns):
        print(f"\n{'='*20} TURN {turn + 1} {'='*20}")
        
        # 1. Generate new flights for this turn
        raw_flights = generate_random_flight_batch(n=4)
        
        # 2. Get the prior state (which is the posterior of the previous turn!)
        prior_weights = assistant.get_expected_weights()
        
        # 3. User makes a choice based on their static ground truth
        choice_idx, norm_features, probabilities = traveler.evaluate_and_choose(raw_flights)
        
        # 4. Assistant updates based on the new observation
        assistant.update_belief_state(raw_flights, choice_idx)
        
        # 5. Get the new posterior state
        posterior_weights = assistant.get_expected_weights()
        
        # 6. Generate the text
        reasoning = generate_reasoning(
            raw_flights, norm_features, choice_idx, prior_weights, posterior_weights,
            assistant.features
        )
        
        # --- Print Formatting ---
        print("AVAILABLE FLIGHTS:")
        for i, f in enumerate(raw_flights):
            price, time, dur, stops = f
            hours = int(time // 60)
            mins = int(time % 60)
            dur_h = int(dur // 60)
            dur_m = int(dur % 60)
            print(f"  [{i}] Price: ${price:.0f} | Dep: {hours:02d}:{mins:02d} | Dur: {dur_h}h {dur_m}m | Stops: {stops}")
        
        print(f"\nUser chose Flight {choice_idx} (Probability of this choice: {probabilities[choice_idx]*100:.1f}%)")
        
        print("\nBELIEF SHIFT (Expected Weights):")
        print(f"  Prior:     {np.round(prior_weights, 3)}")
        print(f"  Posterior: {np.round(posterior_weights, 3)}")
        
        print("\nGENERATED CHAIN-OF-THOUGHT REASONING:")
        print("-" * 50)
        print(reasoning)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    run_sequential_diagnostic(num_turns=4)