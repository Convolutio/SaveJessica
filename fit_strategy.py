from typing import cast
from api_client import SphinxAPIClient
from strategy import MortyRescueStrategy, run_strategy
from sjlearn.planets_simulator import PlanetsBehavior
import pandas as pd

class FitStrategy(MortyRescueStrategy):
    def __init__(self, client: SphinxAPIClient):
        super().__init__(client)
        self.planet_model = PlanetsBehavior()
        self.planet_model.reset()
        self.current_morty_number_in_jessica_planet = 0

    def explore_phase(self, trips_per_planet: int = 75) -> pd.DataFrame:
        """
        Initial exploration phase to understand planet behaviors.
        
        Args:
            trips_per_planet: Total number of trips to send to all the planet
            
        Returns:
            DataFrame with exploration data
        """
        print("\n=== EXPLORATION PHASE ===")
        df = pd.DataFrame([])
        total_number_of_trips = trips_per_planet
        for _ in range(total_number_of_trips):
            # send on the best planet according to our model
            best_planet, _ = self.planet_model.select_best_estimated_planet()
            df = self.collector.explore_planet(best_planet, 1)
            # Check the move
            status = self.client.get_status()
            morties_on_planet_jessica = cast(int, status['morties_on_planet_jessica'])
            reward = morties_on_planet_jessica - self.current_morty_number_in_jessica_planet
            self.current_morty_number_in_jessica_planet = morties_on_planet_jessica
            survived = reward > 0
            # update the estimated phasis of the model
            self.planet_model.declare_step(best_planet, 1, survived)
        self.exploration_data = df
        return df


    def execute_strategy(self):
        print("\n=== EXECUTING ADAPTIVE STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        def get_best_action():
            current_planet, current_planet_survival_rate = self.planet_model.select_best_estimated_planet()
            current_planet_name = self.client.get_planet_name(current_planet)
            morties_per_trip = 3 if current_planet_survival_rate >= 0.8 else (
                2 if current_planet_survival_rate >= 0.7 else 1
            )
            return current_planet, current_planet_name, morties_per_trip

        # Initial best planet
        current_planet, current_planet_name, morties_per_trip = get_best_action()
        
        print(f"Starting with planet: {current_planet_name}")
        
        trips_since_evaluation = 0
        total_trips = 0
        recent_results = []
        
        while morties_remaining > 0:
            # Send Morties
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(current_planet, morties_to_send)
            self.planet_model.declare_step(current_planet, morties_per_trip,
                                           result['survived'] > 0,
                                           skip_phase_estimation=True)
            
            # Track recent results
            recent_results.append({
                'planet': current_planet,
                'survived': result['survived']
            })
            
            morties_remaining = result['morties_in_citadel']
            trips_since_evaluation += 1
            total_trips += 1
            
            # Re-evaluate at each step
            current_planet, current_planet_name, morties_per_trip = get_best_action()

            if total_trips % 50 == 0:
                print(f"  Progress: {total_trips} trips, "
                        f"{result['morties_on_planet_jessica']} saved")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

if __name__ == "__main__":
    print("Morty Express Challenge - Strategy Module")
    print("="*60)
    
    print("\nCurrent strategy:")
    print("1. FitSrategy - Monitor and fit to modeled conditions")
    
    # Uncomment to run:
    run_strategy(FitStrategy, explore_trips=75)
