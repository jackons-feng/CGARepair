"""
Convert JSON localization results to TXT format for backdoor_moo_2.py

Usage: python read.py <json_file> <output_dir> <model_path>
Example: python read.py shapley_backdoor_512_1000_topNone.json ./loc_result/gtsrb_resnet ../model/gtsrb_resnet.h5
"""

import json
import os


def get_layer_index_mapping(model):
    """
    Get mapping from actual layer index to sequential layer index
    
    This converts from the layer indices used by loc_shaply_test.py (actual model.layers indices)
    to the sequential indices expected by xxx_moo_2.py (0, 1, 2, ... for each layer type)
    
    Returns:
        layer_to_seq: dict mapping (layer_type, actual_layer_idx) to sequential index
    """
    layer_to_seq = {}
    conv_count = 0
    dense_count = 0
    
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        lname = type(layer).__name__
        
        if 'Conv' in lname:
            layer_to_seq[('conv', i)] = conv_count
            conv_count += 1
        elif 'Dense' in lname:
            layer_to_seq[('dense', i)] = dense_count
            dense_count += 1
    
    return layer_to_seq


def convert_player_name(player, layer_to_seq):
    """
    Convert player name from actual layer index to sequential index
    
    Args:
        player: e.g., 'conv_5_10' (actual layer 5, neuron 10)
        layer_to_seq: mapping from (layer_type, actual_layer_idx) to sequential index
    
    Returns:
        converted player name, e.g., 'conv_2_10' (2nd conv layer, neuron 10)
    """
    parts = player.split('_')
    if len(parts) != 3:
        return player
    
    layer_type = parts[0]  # 'conv' or 'dense'
    actual_layer_idx = int(parts[1])  # actual layer index in model.layers
    neuron_idx = parts[2]  # neuron index
    
    key = (layer_type, actual_layer_idx)
    if key in layer_to_seq:
        sequential_idx = layer_to_seq[key]
        return f'{layer_type}_{sequential_idx}_{neuron_idx}'
    else:
        print(f"Warning: Layer {layer_type}_{actual_layer_idx} not found in mapping")
        return player


def convert_json_to_txt(json_file, output_dir, model_path=None):
    """
    Convert JSON localization results to TXT format
    
    Args:
        json_file: Path to JSON file containing localization results
        output_dir: Directory to save the TXT files
        model_path: Path to model file (not used, kept for backward compatibility)
    """
    # Load JSON data
    print(f"Loading JSON file: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract players and scores
    players = data['players']
    scores = data['scores']
    
    print(f"Found {len(players)} players and {len(scores)} scores")
    
    # Verify data consistency
    if len(players) != len(scores):
        print(f"Warning: Number of players ({len(players)}) != number of scores ({len(scores)})")
        min_len = min(len(players), len(scores))
        players = players[:min_len]
        scores = scores[:min_len]
        print(f"Truncated to {min_len} entries")
    
    # Note: JSON file already uses sequential layer indices (conv_0, conv_1, ..., conv_19, dense_0)
    # No conversion needed - these match the format expected by backdoor_moo_2.py
    print("\nUsing player names directly from JSON (already in correct sequential format)")
    
    # Show some examples
    print("\nExample players:")
    for i in range(min(10, len(players))):
        print(f"  {players[i]}: {scores[i]}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write players.txt (comma-separated, no quotes)
    players_file = os.path.join(output_dir, 'players.txt')
    print(f"\nWriting players to: {players_file}")
    with open(players_file, 'w') as f:
        f.write(','.join(players))
    
    # Write adv_vals.txt (comma-separated scores)
    adv_vals_file = os.path.join(output_dir, 'adv_vals.txt')
    print(f"Writing adv values to: {adv_vals_file}")
    with open(adv_vals_file, 'w') as f:
        f.write(','.join(map(str, scores)))
    
    print(f"\nConversion complete!")
    print(f"  Players: {len(players)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created:")
    print(f"    - players.txt")
    print(f"    - adv_vals.txt")


if __name__ == '__main__':
    import sys
    
    # Check arguments
    if len(sys.argv) != 4:
        print("Usage: python read.py <model_name> <epsilon> <seed>")
        print("Example: python read.py mnist 0.05 42")
        sys.exit(1)
    
    model_name = sys.argv[1]
    epsilon = sys.argv[2]
    seed = sys.argv[3]
    
    # Construct file paths
    json_file = os.path.join('..', 'rq1', 'result', 'loc_result', model_name, 
                            f'cfrepair_adv_{epsilon}_512_alpha0_steps30_seed{seed}_topNone.json')
    output_dir = os.path.join('loc_result', model_name, f'{epsilon}')
    # Check if input file exists
    if not os.path.exists(json_file):
        print(f"Error: Input file not found: {json_file}")
        sys.exit(1)
    
    try:
        convert_json_to_txt(json_file, output_dir)
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
