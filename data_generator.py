import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random
from enum import Enum

class HandStrength(Enum):
    NUTS = 0
    STRONG = 1
    MARGINAL = 2
    BLUFF = 3

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_SMALL = 3
    BET_MEDIUM = 4
    BET_LARGE = 5
    RAISE_SMALL = 6
    RAISE_MEDIUM = 7
    RAISE_LARGE = 8
    ALL_IN = 9

class PokerDataGenerator:
    def __init__(self, num_hands=10000):
        self.num_hands = num_hands
        self.hand_strengths = list(HandStrength)
        self.actions = list(Action)
        
    def generate_hand_strength(self) -> HandStrength:
        """Generate hand strength based on realistic poker probabilities"""
        # Nuts: 5%, Strong: 20%, Marginal: 45%, Bluff: 30%
        weights = [0.05, 0.20, 0.45, 0.30]
        return np.random.choice(self.hand_strengths, p=weights)
    
    def generate_action_sequence(self, hand_strength: HandStrength, street: str) -> List[Action]:
        """Generate realistic action sequence based on hand strength and street"""
        actions = []
        
        # Pre-flop actions
        if street == "preflop":
            if hand_strength == HandStrength.NUTS:
                # Nuts hands often raise aggressively
                actions = [Action.RAISE_MEDIUM, Action.RAISE_LARGE, Action.ALL_IN]
            elif hand_strength == HandStrength.STRONG:
                # Strong hands raise but not as aggressively
                actions = [Action.RAISE_SMALL, Action.RAISE_MEDIUM, Action.CALL]
            elif hand_strength == HandStrength.MARGINAL:
                # Marginal hands call or fold
                actions = [Action.CALL, Action.FOLD, Action.CHECK]
            else:  # BLUFF
                # Bluffs are mixed - some aggressive, some passive
                actions = [Action.RAISE_SMALL, Action.FOLD, Action.CALL]
        
        # Post-flop actions
        else:
            if hand_strength == HandStrength.NUTS:
                # Nuts hands bet for value
                actions = [Action.BET_MEDIUM, Action.BET_LARGE, Action.ALL_IN]
            elif hand_strength == HandStrength.STRONG:
                # Strong hands bet but can be cautious
                actions = [Action.BET_SMALL, Action.BET_MEDIUM, Action.CHECK]
            elif hand_strength == HandStrength.MARGINAL:
                # Marginal hands check-call or fold
                actions = [Action.CHECK, Action.CALL, Action.FOLD]
            else:  # BLUFF
                # Bluffs can be aggressive or passive
                actions = [Action.BET_SMALL, Action.CHECK, Action.FOLD]
        
        # Add some randomness to make it more realistic
        sequence_length = random.randint(1, 4)
        return random.choices(actions, k=sequence_length)
    
    def encode_action_sequence(self, actions: List[Action], max_length: int = 10) -> List[int]:
        """Encode action sequence to fixed-length vector"""
        encoded = [action.value for action in actions]
        
        # Pad or truncate to max_length
        if len(encoded) < max_length:
            encoded.extend([-1] * (max_length - len(encoded)))  # -1 for padding
        else:
            encoded = encoded[:max_length]
            
        return encoded
    
    def generate_street_features(self, street: str) -> Dict[str, int]:
        """Generate features for different streets"""
        street_encoding = {
            "preflop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3
        }
        
        return {
            "street": street_encoding[street],
            "position": random.randint(0, 8),  # 0=UTG, 8=BB
            "pot_size": random.randint(10, 1000),
            "stack_size": random.randint(100, 5000),
            "num_players": random.randint(2, 9)
        }
    
    def generate_single_hand(self) -> Dict:
        """Generate a single hand with all features"""
        hand_strength = self.generate_hand_strength()
        streets = ["preflop", "flop", "turn", "river"]
        
        # Generate action sequences for each street
        all_actions = []
        street_features = {}
        
        for street in streets:
            actions = self.generate_action_sequence(hand_strength, street)
            encoded_actions = self.encode_action_sequence(actions)
            all_actions.extend(encoded_actions)
            
            # Add street-specific features
            street_feats = self.generate_street_features(street)
            for key, value in street_feats.items():
                street_features[f"{street}_{key}"] = value
        
        return {
            "hand_strength": hand_strength.value,
            "actions": all_actions,
            **street_features
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        data = []
        
        for _ in range(self.num_hands):
            hand_data = self.generate_single_hand()
            data.append(hand_data)
        
        df = pd.DataFrame(data)
        
        # Create action columns
        for i in range(40):  # 10 actions per street * 4 streets
            df[f"action_{i}"] = df["actions"].apply(lambda x: x[i] if i < len(x) else -1)
        
        # Drop the original actions column
        df = df.drop("actions", axis=1)
        
        return df
    
    def save_dataset(self, filename: str = "poker_range_data.csv"):
        """Generate and save the dataset"""
        print(f"Generating {self.num_hands} poker hands...")
        df = self.generate_dataset()
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        print(f"Hand strength distribution:\n{df['hand_strength'].value_counts().sort_index()}")
        return df

if __name__ == "__main__":
    generator = PokerDataGenerator(num_hands=10000)
    df = generator.save_dataset()
