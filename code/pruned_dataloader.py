"""
Enhanced Pruned Dataset with ID Remapping for LightGCN
Eliminates zombie users/items and provides seamless integration
"""
import os
import torch
import numpy as np
import pandas as pd
from dataloader import BasicDataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

def add_id_remapping_to_pruned_data(pruned_train_path, submission_path, test_path, output_dir):
    """
    Add ID remapping to existing pruned data from gp.py
    The pruned training should already have all test users AND items preserved.
    
    Args:
        pruned_train_path: Path to pruned train.csv from gp.py (already has test users+items preserved)
        submission_path: Path to original sample_submission.csv
        test_path: Path to original test.csv
        output_dir: Directory to save remapped files and mappings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data for ID remapping...")
    pruned_train_df = pd.read_csv(pruned_train_path)
    submission_df = pd.read_csv(submission_path)
    original_test_df = pd.read_csv(test_path)
    
    print(f"Pruned training interactions: {len(pruned_train_df)}")
    print(f"Original test interactions: {len(original_test_df)}")
    
    # Create mappings based on pruned training data
    # (which should already include all test users AND items)
    unique_users = sorted(pruned_train_df['user_id'].unique())
    unique_items = sorted(pruned_train_df['item_id'].unique())
    
    print(f"Creating mappings based on pruned training data:")
    print(f"  Users: {len(unique_users)}")
    print(f"  Items: {len(unique_items)}")
    
    # Verify test coverage before creating mappings
    test_users_in_original = set(original_test_df['user_id'].unique())
    test_items_in_original = set(original_test_df['item_id'].unique())
    training_users_in_pruned = set(pruned_train_df['user_id'].unique())
    training_items_in_pruned = set(pruned_train_df['item_id'].unique())
    
    user_overlap = test_users_in_original.intersection(training_users_in_pruned)
    item_overlap = test_items_in_original.intersection(training_items_in_pruned)
    
    print(f"\nTest Coverage Analysis:")
    print(f"Test users in original: {len(test_users_in_original)}")
    print(f"Test users preserved in training: {len(user_overlap)}/{len(test_users_in_original)} ({len(user_overlap)/len(test_users_in_original)*100:.1f}%)")
    print(f"Test items in original: {len(test_items_in_original)}")
    print(f"Test items preserved in training: {len(item_overlap)}/{len(test_items_in_original)} ({len(item_overlap)/len(test_items_in_original)*100:.1f}%)")
    
    
    # Create bidirectional mappings
    old_to_new_user = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    new_to_old_user = {new_id: old_id for old_id, new_id in old_to_new_user.items()}
    
    old_to_new_item = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    new_to_old_item = {new_id: old_id for old_id, new_id in old_to_new_item.items()}
    
    # Apply remapping to pruned training dataframe
    remapped_train = pruned_train_df.copy()
    
    # Store original timestamp before any operations
    original_train_timestamp = None
    if 'timestamp' in remapped_train.columns:
        original_train_timestamp = remapped_train['timestamp'].copy()
    
    # Only remap user_id and item_id
    remapped_train['user_id'] = remapped_train['user_id'].map(old_to_new_user).astype(int)
    remapped_train['item_id'] = remapped_train['item_id'].map(old_to_new_item).astype(int)
    
    # Restore original timestamp values
    if original_train_timestamp is not None:
        remapped_train['timestamp'] = original_train_timestamp.astype(int)
    
    # Apply remapping to submission file
    remapped_submission = submission_df.copy()
    # Check which submission users can be mapped
    mappable_submission_users = submission_df['user_id'].isin(old_to_new_user.keys())
    print(f"\nSubmission users mappable: {mappable_submission_users.sum()}/{len(submission_df)}")
    
    remapped_submission['user_id'] = remapped_submission['user_id'].map(old_to_new_user)
    remapped_submission = remapped_submission.dropna()
    remapped_submission['user_id'] = remapped_submission['user_id'].astype(int)
    
    # Apply remapping to test set
    remapped_test = original_test_df.copy()
    
    # Store original timestamp before any operations
    original_test_timestamp = None
    if 'timestamp' in remapped_test.columns:
        original_test_timestamp = remapped_test['timestamp'].copy()
    
    # Check mappability BEFORE applying mappings
    test_users_mappable = original_test_df['user_id'].isin(old_to_new_user.keys())
    test_items_mappable = original_test_df['item_id'].isin(old_to_new_item.keys())
    
    print(f"\nDetailed Test Mappability:")
    print(f"Test users mappable: {test_users_mappable.sum()}/{len(original_test_df)} ({test_users_mappable.mean()*100:.1f}%)")
    print(f"Test items mappable: {test_items_mappable.sum()}/{len(original_test_df)} ({test_items_mappable.mean()*100:.1f}%)")
    
    # Only remap user_id and item_id
    remapped_test['user_id'] = remapped_test['user_id'].map(old_to_new_user)
    remapped_test['item_id'] = remapped_test['item_id'].map(old_to_new_item)
    
    # Remove unmappable entries (where user_id or item_id became NaN)
    test_before_filter = len(remapped_test)
    valid_rows = remapped_test['user_id'].notna() & remapped_test['item_id'].notna()
    remapped_test = remapped_test[valid_rows].copy()
    test_after_filter = len(remapped_test)
    
    # Convert IDs to integers
    remapped_test['user_id'] = remapped_test['user_id'].astype(int)
    remapped_test['item_id'] = remapped_test['item_id'].astype(int)
    
    # Restore original timestamp values for the remaining rows
    if original_test_timestamp is not None:
        remapped_test['timestamp'] = original_test_timestamp[valid_rows].astype(int)
    
    print(f"Test interactions after filtering: {test_after_filter}/{len(original_test_df)} ({test_after_filter/len(original_test_df)*100:.1f}%)")
    
    # Save files
    train_output = os.path.join(output_dir, 'train_remapped.csv')
    submission_output = os.path.join(output_dir, 'sample_submission_remapped.csv')
    test_output = os.path.join(output_dir, 'test_remapped.csv')
    
    remapped_train.to_csv(train_output, index=False)
    remapped_submission.to_csv(submission_output, index=False)
    remapped_test.to_csv(test_output, index=False)
    
    print(f"\nFiles saved:")
    print(f"Remapped training data: {train_output}")
    print(f"Remapped submission file: {submission_output}")
    print(f"Remapped test data: {test_output}")
    
    # Calculate memory savings
    try:
        original_matrix_size = (max(unique_users) + 1) * (max(unique_items) + 1)
        remapped_matrix_size = len(unique_users) * len(unique_items)
        compression_ratio = original_matrix_size / remapped_matrix_size
        
        print(f"\nID Remapping Results:")
        print(f"Matrix compression: {compression_ratio:.1f}x")
        print(f"Users: {max(unique_users)+1} -> {len(unique_users)}")
        print(f"Items: {max(unique_items)+1} -> {len(unique_items)}")
    except Exception as e:
        print(f"Error calculating matrix compression: {e}")
        print(f"Users: {len(unique_users)}")
        print(f"Items: {len(unique_items)}")
    
    return {
        'old_to_new_user': old_to_new_user,
        'new_to_old_user': new_to_old_user,
        'old_to_new_item': old_to_new_item,
        'new_to_old_item': new_to_old_item,
        'n_users': len(unique_users),
        'n_items': len(unique_items),
        'test_coverage': {
            'user_coverage': len(user_overlap) / len(test_users_in_original),
            'item_coverage': len(item_overlap) / len(test_items_in_original),
            'interaction_coverage': test_after_filter / len(original_test_df)
        }
    }

# Rest of the PrunedDataset class remains the same...
class PrunedDataset(BasicDataset):
    """
    Drop-in replacement for AmazonBeauty dataset using pruned and remapped data
    Compatible with existing LightGCN code
    """
    def __init__(self, data_dir, train_file='train_remapped.csv', test_file='test_remapped.csv', submission_file='sample_submission_remapped.csv'):
        cprint("Loading [PrunedDataset] with ID remapping")
        self.data_dir = data_dir
        
        # Load remapped training data
        train_path = os.path.join(data_dir, train_file)
        self.train_df = pd.read_csv(train_path)
        
        # Load remapped test data
        test_path = os.path.join(data_dir, test_file)
        self.test_df = pd.read_csv(test_path)
        
        # Load remapped submission data
        submission_path = os.path.join(data_dir, submission_file)
        self.submission_df = pd.read_csv(submission_path)
        
        # Extract arrays
        self.train_user_ids = self.train_df['user_id'].values
        self.train_item_ids = self.train_df['item_id'].values
        self.test_user_ids = self.test_df['user_id'].values
        self.test_item_ids = self.test_df['item_id'].values
        
        # Set dimensions (now using compact 0-based IDs)
        self.n_user = max(self.train_df['user_id'].max(), self.test_df['user_id'].max()) + 1
        self.m_item = max(self.train_df['item_id'].max(), self.test_df['item_id'].max()) + 1
        self.train_data_size = len(self.train_df)
        self.test_data_size = len(self.test_df)
        
        # Get unique users for iteration
        self.train_unique_users = np.unique(self.train_user_ids)
        self.test_unique_users = np.unique(self.test_user_ids)
        
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")
        print(f"{self.n_user} users, {self.m_item} items")
        print(f"Sparsity: {(self.train_data_size + self.test_data_size) / (self.n_user * self.m_item):.6f}")
        
        # Create user-item interaction matrix (now dense ID space!)
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_user_ids)), (self.train_user_ids, self.train_item_ids)),
            shape=(self.n_user, self.m_item)
        )
        
        # Calculate user and item degrees
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        
        # Pre-calculate positive items
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        
        # Build test dict from actual test data
        self._testDict = self._build_test_dict()
        
        # Initialize graph
        self.Graph = None
        
        # Load ID mappings if available
        self.id_mappings = None
        try:
            mappings_path = os.path.join(data_dir, 'id_mappings.pkl')
            if os.path.exists(mappings_path):
                import pickle
                with open(mappings_path, 'rb') as f:
                    self.id_mappings = pickle.load(f)
                print("ID mappings loaded successfully")
                
                # Print test coverage information if available
                if 'test_coverage' in self.id_mappings:
                    coverage = self.id_mappings['test_coverage']
                    print(f"Test coverage: Users {coverage['user_coverage']*100:.1f}%, Items {coverage['item_coverage']*100:.1f}%, Interactions {coverage['interaction_coverage']*100:.1f}%")
        except:
            print("No ID mappings found - will need to provide manually if generating submissions")
        
        print("PrunedDataset is ready to go")
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.train_data_size
    
    @property
    def allPos(self):
        return self._allPos
    
    @property
    def testDict(self):
        return self._testDict
    
    def set_id_mappings(self, mappings):
        """Set ID mappings manually if not loaded from file"""
        self.id_mappings = mappings
        print("ID mappings set manually")
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert scipy sparse matrix to torch sparse tensor"""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """Build the sparse graph for GCN"""
        print("Loading adjacency matrix")
        if self.Graph is None:
            try:
                # Try to load pre-computed adjacency matrix
                adj_path = os.path.join(self.data_dir, 's_pre_adj_mat_pruned.npz')
                pre_adj_mat = sp.load_npz(adj_path)
                print("Successfully loaded pre-computed adjacency matrix")
                norm_adj = pre_adj_mat
            except:
                # Compute adjacency matrix from scratch
                print("Generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items), 
                    dtype=np.float32
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                
                # Build bipartite graph
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                
                end = time()
                print(f"Adjacency matrix generated in {end-s:.2f}s")
                
                # Save the computed adjacency matrix
                sp.save_npz(adj_path, norm_adj)
            
            # Convert to torch sparse tensor
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            print("Graph built and ready")
        
        return self.Graph
    
    def getUserItemFeedback(self, users, items):
        """Get user-item feedback"""
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
    
    def getUserPosItems(self, users):
        """Get positive items for each user"""
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def _build_test_dict(self):
        """
        Build test dictionary from actual test data
        """
        test_dict = {}
        for i, item in enumerate(self.test_item_ids):
            user = self.test_user_ids[i]
            if test_dict.get(user):
                test_dict[user].append(item)
            else:
                test_dict[user] = [item]
        return test_dict
    
    def get_submission_recommendations(self, model, top_k=10, output_file='submission.csv'):
        """
        Generate recommendations for submission file
        Automatically handles ID remapping back to original space
        """
        if self.id_mappings is None:
            raise ValueError("ID mappings not available. Cannot convert back to original IDs.")
        
        print("Generating submission recommendations...")
        results = []
        model.eval()
        
        with torch.no_grad():
            for idx, row in self.submission_df.iterrows():
                new_user_id = row['user_id']
                original_user_id = self.id_mappings['new_to_old_user'][new_user_id]
                
                # Get user ratings
                user_tensor = torch.LongTensor([new_user_id]).to(world.device)
                rating = model.getUsersRating(user_tensor)
                rating = rating.cpu().numpy()[0]
                
                # Exclude items the user has already interacted with
                user_pos_items = self.getUserPosItems([new_user_id])[0]
                rating[user_pos_items] = -float('inf')
                
                # Get top-k items (in new ID space)
                top_new_item_ids = np.argsort(-rating)[:top_k]
                
                # Convert back to original item IDs
                top_original_item_ids = [
                    self.id_mappings['new_to_old_item'][item_id] 
                    for item_id in top_new_item_ids 
                    if item_id in self.id_mappings['new_to_old_item']
                ]
                
                # Format recommendations
                rec_str = ','.join(map(str, top_original_item_ids))
                
                results.append({
                    'ID': row['ID'],
                    'user_id': original_user_id,
                    'item_id': rec_str
                })
        
        # Save submission file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        return results_df
    
    def __getitem__(self, index):
        """For dataset indexing"""
        user = self.train_unique_users[index]
        return user
    
    def __len__(self):
        """Dataset length"""
        return len(self.train_unique_users)

# Enhanced pipeline function with better validation
def create_pruned_dataset(train_path, test_path, submission_path, output_dir, strategy='smart_budget', target_memory_gb=30):
    """
    Complete pipeline: prune + remap + create dataset
    Enhanced with test item preservation validation
    """
    import gp
    
    print("Step 1: Running graph pruning...")
    pruned_path = os.path.join(output_dir, f'train_{strategy}.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run enhanced pruning that preserves test items
    gp.prune_graph_for_lightgcn(
        train_path=train_path,
        submission_path=submission_path,
        test_path=test_path, 
        output_path=pruned_path,
        strategy=strategy,
        target_memory_gb=target_memory_gb
    )
    
    print("\nStep 2: Adding ID remapping with test validation...")
    mappings = add_id_remapping_to_pruned_data(
        pruned_train_path=pruned_path,
        submission_path=submission_path,
        test_path=test_path,
        output_dir=output_dir
    )
    
    # Validate test coverage
    if 'test_coverage' in mappings:
        coverage = mappings['test_coverage']
        print(f"\n Test Coverage Summary:")
        print(f"  User coverage: {coverage['user_coverage']*100:.1f}%")
        print(f"  Item coverage: {coverage['item_coverage']*100:.1f}%")
        print(f"  Interaction coverage: {coverage['interaction_coverage']*100:.1f}%")
        
    
    # Save mappings
    import pickle
    mappings_path = os.path.join(output_dir, 'id_mappings.pkl')
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"ID mappings saved: {mappings_path}")
    
    print("\nStep 3: Creating dataset...")
    dataset = PrunedDataset(output_dir)
    dataset.set_id_mappings(mappings)
    
    
    # Final validation
    print(f"\n Final Validation:")
    print(f"  Training interactions: {len(dataset.train_df):,}")
    print(f"  Test interactions: {len(dataset.test_df):,}")
    print(f"  Users: {dataset.n_users:,}")
    print(f"  Items: {dataset.m_items:,}")
    print(f"  Submission users: {len(dataset.submission_df):,}")
    
    return dataset

# Usage example
if __name__ == "__main__":
    # Complete pipeline with enhanced test item preservation
    dataset = create_pruned_dataset(
        train_path='../data/beauty/train.csv',
        test_path='../data/beauty/test.csv',
        submission_path='../data/beauty/sample_submission.csv',
        output_dir='../data/beauty/pruned_data',
        strategy='smart_budget',
        target_memory_gb=40  
    )