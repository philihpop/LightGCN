import pandas as pd
import numpy as np
from collections import Counter
import os

def prune_graph_for_lightgcn(train_path, submission_path, output_path, 
                           strategy='smart_budget', target_memory_gb=30, min_user_interactions=1,
                           test_path=None):
    """
    Prune the training graph while preserving target users and relevant collaborative signals
    
    Args:
        train_path: Path to train.csv
        submission_path: Path to sample_submission.csv  
        output_path: Path to save pruned train.csv
        strategy: 'conservative', 'moderate', 'aggressive', or 'smart_budget'
        target_memory_gb: Target memory usage for smart_budget strategy (default: 30)
        min_user_interactions: Minimum interactions for conservative strategy (default: 1)
        test_path: Path to test.csv (to preserve test users AND items in training)
    """
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    submission_df = pd.read_csv(submission_path)
    
    # Get target users from submission
    target_users = set(submission_df['user_id'].values)
    print(f"Target users to preserve: {len(target_users)}")
    
    # Get test users AND items to preserve in training
    test_users = set()
    test_items = set()
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_users = set(test_df['user_id'].values)
        test_items = set(test_df['item_id'].values)
        print(f"Test users to preserve in training: {len(test_users)}")
        print(f"Test items to preserve in training: {len(test_items)}")
    
    # Combine target and test users - these MUST be kept in training
    users_to_preserve = target_users.union(test_users)
    print(f"Total users to preserve in training: {len(users_to_preserve)}")
    
    # Find items that target users interact with
    target_user_items = set(train_df[train_df['user_id'].isin(target_users)]['item_id'].values)
    print(f"Items connected to target users: {len(target_user_items)}")
    
    # Combine target items and test items - these should be prioritized
    priority_items = target_user_items.union(test_items)
    print(f"Priority items (target + test): {len(priority_items)}")
    
    # Calculate user interaction counts
    user_counts = train_df['user_id'].value_counts().to_dict()
    
    # Apply different pruning strategies - but ALWAYS preserve target + test users AND test items
    if strategy == 'conservative':
        # Keep preserved users + users with ≥min_user_interactions on priority items
        users_to_keep = set(users_to_preserve)  # ALWAYS keep these
        priority_item_interactions = train_df[train_df['item_id'].isin(priority_items)]
        
        for user_id in priority_item_interactions['user_id'].unique():
            if user_id not in users_to_preserve and user_counts[user_id] >= min_user_interactions:
                users_to_keep.add(user_id)
        
        # Keep all interactions with priority items + all interactions of preserved users
        pruned_df = train_df[
            train_df['user_id'].isin(users_to_keep) | 
            train_df['item_id'].isin(priority_items)
        ].copy()
        
    elif strategy == 'moderate':
        # Keep preserved users + ALL users who interact with priority items
        priority_item_users = set(train_df[train_df['item_id'].isin(priority_items)]['user_id'].values)
        users_to_keep = users_to_preserve.union(priority_item_users)
        
        # Keep all interactions with priority items + all interactions of selected users
        pruned_df = train_df[
            train_df['user_id'].isin(users_to_keep) | 
            train_df['item_id'].isin(priority_items)
        ].copy()
        
    elif strategy == 'aggressive':
        # Keep ALL interactions with priority items + preserved user interactions
        pruned_df = train_df[
            train_df['user_id'].isin(users_to_preserve) | 
            train_df['item_id'].isin(priority_items)
        ].copy()
        
    elif strategy == 'smart_budget':
        # Budget-based approach: prioritize most valuable interactions
        max_interactions = int(len(train_df) * (target_memory_gb / 157))
        print(f"Budget: {max_interactions:,} interactions (~{target_memory_gb}GB)")
        
        # Priority 1: All preserved user interactions (MUST keep)
        preserved_interactions = train_df[train_df['user_id'].isin(users_to_preserve)]
        print(f"Reserved for preserved users: {len(preserved_interactions)} interactions")
        
        # Priority 2: All interactions with test items (MUST keep for test evaluation)
        test_item_interactions = train_df[train_df['item_id'].isin(test_items)]
        print(f"Reserved for test items: {len(test_item_interactions)} interactions")
        
        # Combine priority 1 and 2 (remove duplicates)
        essential_interactions = train_df[
            train_df['user_id'].isin(users_to_preserve) | 
            train_df['item_id'].isin(test_items)
        ]
        print(f"Essential interactions (users + test items): {len(essential_interactions)}")
        
        # Priority 3: Non-preserved users ranked by engagement with target items
        remaining_train = train_df[
            (~train_df['user_id'].isin(users_to_preserve)) & 
            (~train_df['item_id'].isin(test_items)) 
        ]
        
        # Calculate engagement scores for remaining users
        user_engagement = {}
        for _, row in remaining_train.iterrows():
            user_id = row['user_id']
            if user_id not in user_engagement:
                user_engagement[user_id] = {
                    'target_item_interactions': 0,
                    'total_interactions': user_counts[user_id]
                }
            user_engagement[user_id]['target_item_interactions'] += 1
        
        # Score users by engagement (target interactions * log(total interactions))
        user_scores = []
        for user_id, stats in user_engagement.items():
            score = stats['target_item_interactions'] * np.log(stats['total_interactions'] + 1)
            user_scores.append((user_id, score, stats['total_interactions']))
        
        # Sort by score descending
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select users within budget
        selected_users = set(users_to_preserve)  # ALWAYS include preserved users
        remaining_budget = max_interactions - len(essential_interactions)
        print(f"Remaining budget for additional users: {remaining_budget}")
        
        for user_id, score, total_interactions in user_scores:
            if total_interactions <= remaining_budget:
                selected_users.add(user_id)
                remaining_budget -= total_interactions
        
        # Final selection: essential interactions + selected user interactions
        pruned_df = train_df[
            train_df['user_id'].isin(selected_users) | 
            train_df['item_id'].isin(test_items)  # Ensure all test items are preserved
        ].copy()
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print("Filtering interactions...")
    original_size = len(train_df)
    
    # Remove items that have no interactions after user pruning (but keep test items!)
    item_counts_after_pruning = pruned_df['item_id'].value_counts()
    items_to_keep = set(item_counts_after_pruning.index)
    
    # Ensure test items are always kept even if they have no interactions in pruned data
    items_to_keep = items_to_keep.union(test_items)
    
    # Only filter out items that are not test items and have no interactions
    pruned_df = pruned_df[pruned_df['item_id'].isin(items_to_keep)]
    
    # Final statistics
    final_size = len(pruned_df)
    unique_users = pruned_df['user_id'].nunique()
    unique_items = pruned_df['item_id'].nunique()
    
    print(f"\n=== PRUNING RESULTS ({strategy.upper()}) ===")
    print(f"Original interactions: {original_size:,}")
    print(f"Pruned interactions: {final_size:,}")
    print(f"Reduction: {(1 - final_size/original_size)*100:.1f}%")
    print(f"")
    print(f"Original users: {train_df['user_id'].nunique():,}")
    print(f"Remaining users: {unique_users:,}")
    print(f"User reduction: {(1 - unique_users/train_df['user_id'].nunique())*100:.1f}%")
    print(f"")
    print(f"Original items: {train_df['item_id'].nunique():,}")
    print(f"Remaining items: {unique_items:,}")
    print(f"Item reduction: {(1 - unique_items/train_df['item_id'].nunique())*100:.1f}%")
    
    # Verify all preserved users are kept
    remaining_preserved_users = set(pruned_df['user_id'].values) & users_to_preserve
    print(f"")
    print(f"Target users preserved: {len(remaining_preserved_users & target_users)}/{len(target_users)}")
    if test_users:
        print(f"Test users preserved in training: {len(remaining_preserved_users & test_users)}/{len(test_users)}")
    
    # Check target item coverage
    remaining_target_items = set(pruned_df['item_id'].values) & target_user_items
    print(f"Target items preserved: {len(remaining_target_items)}/{len(target_user_items)}")
    
    # Check test item coverage (CRITICAL)
    if test_items:
        remaining_test_items = set(pruned_df['item_id'].values) & test_items
        print(f"Test items preserved: {len(remaining_test_items)}/{len(test_items)}")
        
        if len(remaining_test_items) != len(test_items):
            missing_test_items = test_items - remaining_test_items
            print(f"❌ WARNING: Missing test items: {len(missing_test_items)}")
            print(f"Missing test items: {list(missing_test_items)[:10]}...")  # Show first 10
        else:
            print(f"✅ All test items successfully preserved in training")
    
    if len(remaining_preserved_users) != len(users_to_preserve):
        missing_users = users_to_preserve - remaining_preserved_users
        print(f"❌ WARNING: Missing preserved users: {len(missing_users)}")
        print(f"Missing users: {list(missing_users)[:10]}...")  # Show first 10
    else:
        print(f"✅ All preserved users successfully kept in training")
    
    # Save pruned dataset
    pruned_df.to_csv(output_path, index=False)
    print(f"")
    print(f"Pruned dataset saved to: {output_path}")
    
    # Memory estimation
    estimated_memory_reduction = (1 - final_size/original_size)
    print(f"")
    print(f"Estimated memory reduction: {estimated_memory_reduction*100:.1f}%")
    print(f"If original model needed 157GB, pruned model needs ~{157*(1-estimated_memory_reduction):.1f}GB")
    
    return pruned_df

# Usage examples
if __name__ == "__main__":
    import os
    
    # Strategy 1: Conservative (~1.3GB) - safest option
    print("=== CONSERVATIVE STRATEGY ===")
    conservative_data = prune_graph_for_lightgcn(
        train_path='../data/beauty/train.csv',
        submission_path='../data/beauty/sample_submission.csv', 
        test_path='../data/beauty/test.csv',  # CRITICAL: Include test path
        output_path='../data/beauty/train_conservative.csv',
        strategy='conservative',
        min_user_interactions=2
    )
    
    # Strategy 2: Smart Budget (~30GB) - recommended for 64GB RAM
    print("\n" + "="*50)
    print("=== SMART BUDGET STRATEGY ===")
    budget_data = prune_graph_for_lightgcn(
        train_path='../data/beauty/train.csv',
        submission_path='../data/beauty/sample_submission.csv',
        test_path='../data/beauty/test.csv',  # CRITICAL: Include test path
        output_path='../data/beauty/train_smart_budget.csv',
        strategy='smart_budget',
        target_memory_gb=30
    )
    
    # Strategy 3: Moderate (~22GB) - good balance
    print("\n" + "="*50)
    print("=== MODERATE STRATEGY ===")
    moderate_data = prune_graph_for_lightgcn(
        train_path='../data/beauty/train.csv',
        submission_path='../data/beauty/sample_submission.csv',
        test_path='../data/beauty/test.csv',  # CRITICAL: Include test path
        output_path='../data/beauty/train_moderate.csv',
        strategy='moderate'
    )
    
    # Strategy 4: Aggressive (~19GB) - maximum collaborative signals
    print("\n" + "="*50)
    print("=== AGGRESSIVE STRATEGY ===")
    aggressive_data = prune_graph_for_lightgcn(
        train_path='../data/beauty/train.csv',
        submission_path='../data/beauty/sample_submission.csv',
        test_path='../data/beauty/test.csv',  
        output_path='../data/beauty/train_aggressive.csv',
        strategy='aggressive'
    )