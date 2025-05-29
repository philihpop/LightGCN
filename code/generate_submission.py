"""
Simple script to generate recommendations for submission
Uses the original parse_args from parse.py and keeps everything as simple as possible
"""
import world
import utils
import torch
import numpy as np
import pandas as pd
import os
from parse import parse_args
import register
from register import dataset
from model import LightGCN, PureMF

# Use the original parse_args to maintain compatibility
args = parse_args()
world.config['latent_dim_rec'] = args.recdim
world.config['lightGCN_n_layers'] = args.layer
world.dataset = args.dataset
world.model_name = args.model
world.LOAD = 1  # Always load the model

# Main function for generating recommendations
def generate_recommendations():
    # Define parameters
    top_k = 10  # Number of recommendations to generate
    batch_size = 50  # Process users in batches to save memory
    submission_file = "../data/beauty/sample_submission.csv"
    output_file = "../data/beauty/submission_results.csv"
    
    print(f"Dataset: {world.dataset}, Model: {world.model_name}")
    print(f"Generating top-{top_k} recommendations from {submission_file}")
    
    # Get model class
    if world.model_name == 'mf':
        model_class = PureMF
    else:
        model_class = LightGCN
    
    # Load trained model
    weight_file = utils.getFileName()
    print(f"Loading model weights from {weight_file}")
    
    # Initialize model
    model = model_class(world.config, dataset)
    model = model.to(world.device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"Successfully loaded model weights from {weight_file}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print(f"Make sure the model file exists at: {weight_file}")
        return
    
    # Load submission file
    try:
        print(f"Loading submission file from: {submission_file}")
        sub_df = pd.read_csv(submission_file)
        user_ids = sub_df['user_id'].values
        print(f"Found {len(user_ids)} users in submission file")
    except Exception as e:
        print(f"Error loading submission file: {e}")
        return
    
    # Set model to eval mode
    model.eval()
    
    # Generate recommendations for each user in batches
    results = []
    with torch.no_grad():
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]
            print(f"Processing users {i+1}-{min(i+batch_size, len(user_ids))} of {len(user_ids)}")
            
            for user in batch_users:
                # Handle users not in training
                if user >= dataset.n_users:
                    print(f"User {user} not in training data, using random recommendations")
                    recommendations = np.random.choice(range(dataset.m_items), size=top_k, replace=False)
                else:
                    # Get user tensor
                    user_tensor = torch.LongTensor([user]).to(world.device)
                    
                    # Get ratings for all items
                    rating = model.getUsersRating(user_tensor)
                    rating = rating.cpu().numpy()
                    
                    # Exclude items the user has already interacted with
                    try:
                        user_pos_items = dataset.getUserPosItems([user])[0]
                        rating[0, user_pos_items] = -float('inf')
                    except:
                        print(f"Warning: Could not get positive items for user {user}")
                    
                    # Get top-k items
                    top_indices = np.argsort(-rating[0])[:top_k]
                    recommendations = top_indices
                
                # Format recommendations as comma-separated string
                rec_str = ','.join(map(str, recommendations))
                
                # Find the ID for this user in the submission file
                user_row = sub_df[sub_df['user_id'] == user]
                if len(user_row) > 0:
                    user_id = user_row['ID'].values[0]
                    results.append({
                        'ID': user_id,
                        'user_id': user,
                        'item_id': rec_str
                    })
                else:
                    print(f"Warning: User {user} not found in submission file")
    
    # Create and save results dataframe
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Recommendations saved to {output_file}")
    else:
        print("No recommendations generated")

if __name__ == '__main__':
    generate_recommendations()