#!/usr/bin/env python3
"""

Part 1: Data loading and preprocessing

"""

import numpy as np

# Initialize dimensions
num_users = 10000
num_movies = 10000

# --------------------------------------------------------
# 1. Build user ID → row index mapping
# --------------------------------------------------------

user_map = {}   # original_user_id → matrix row index

with open("users.txt", "r") as f:
    for idx, line in enumerate(f):
        uid = int(line.strip())
        user_map[uid] = idx   # map to [0..9999]

print("Loaded user mapping:", len(user_map))

# --------------------------------------------------------
# 2. Load rating file function
# --------------------------------------------------------

def load_matrix(filename, X, M=None):
    count = 0
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = int(parts[2])

            # Map user ID to [0..9999]
            if user_id not in user_map:
                continue  # skip if ID not in your subset

            u = user_map[user_id]
            m = movie_id - 1   # movie IDs are already 1..10000

            # Validate indices
            if u < num_users and m < num_movies:
                X[u, m] = rating
                if M is not None:
                    M[u, m] = 1
                count += 1
                
            if count % 1000000 == 0:
                print(f"Processed {count} ratings...")
    
    return count

# --------------------------------------------------------
# 3. Load training data
# --------------------------------------------------------

print("\nLoading training data...")
X_train = np.zeros((num_users, num_movies), dtype=np.uint8)
M_train = np.zeros((num_users, num_movies), dtype=np.uint8)

train_count = load_matrix("netflix_train.txt", X_train, M_train)
print(f"Training data loaded: {train_count} ratings")
print(f"X_train shape: {X_train.shape}")
print(f"Known ratings in training: {M_train.sum()}")

# --------------------------------------------------------
# 4. Load testing data
# --------------------------------------------------------

print("\nLoading testing data...")
X_test = np.zeros((num_users, num_movies), dtype=np.uint8)
M_test = np.zeros((num_users, num_movies), dtype=np.uint8)

test_count = load_matrix("netflix_test.txt", X_test, M_test)
print(f"Test data loaded: {test_count} ratings")
print(f"X_test shape: {X_test.shape}")
print(f"Known ratings in test: {M_test.sum()}")

# --------------------------------------------------------
# 5. Verification and statistics
# --------------------------------------------------------

print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)
print(f"Total training ratings: {train_count}")
print(f"Total test ratings: {test_count}")
print(f"Total ratings: {train_count + test_count}")
print(f"Training matrix density: {M_train.sum() / (num_users * num_movies):.4f}")
print(f"Test matrix density: {M_test.sum() / (num_users * num_movies):.4f}")

# Check for overlap between train and test
overlap_mask = np.logical_and(M_train, M_test)
overlap_count = np.sum(overlap_mask)
print(f"Overlap between train and test sets: {overlap_count} ratings")

# Rating distribution analysis
def analyze_ratings(X, M, name):
    ratings = X[M == 1]
    print(f"\n{name} Rating Distribution:")
    for rating in range(1, 6):
        count = np.sum(ratings == rating)
        percentage = (count / len(ratings)) * 100
        print(f"  Rating {rating}: {count:,} ({percentage:.1f}%)")

analyze_ratings(X_train, M_train, "TRAINING")
analyze_ratings(X_test, M_test, "TEST")

# --------------------------------------------------------
# 6. Sample data inspection
# --------------------------------------------------------

print("\n" + "="*50)
print("SAMPLE DATA INSPECTION")
print("="*50)

# Check first few users
for user_idx in range(3):
    user_ratings = np.sum(M_train[user_idx])
    print(f"User {user_idx}: {user_ratings} ratings in training set")
    
    # Find original user ID
    original_uid = None
    for uid, idx in user_map.items():
        if idx == user_idx:
            original_uid = uid
            break
    print(f"  Original user ID: {original_uid}")

# Check first few movies
for movie_idx in range(3):
    movie_ratings_train = np.sum(M_train[:, movie_idx])
    movie_ratings_test = np.sum(M_test[:, movie_idx])
    print(f"Movie {movie_idx + 1}: {movie_ratings_train} train ratings, {movie_ratings_test} test ratings")


"""

Part 2, 3: Implementation of baselines, user-based collaborative filtering, and matrix factorization

"""

import os
import sys
import time
import math
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Config (safer defaults)
# -----------------------------
DATA_DIR = '.'  
USERS_FILE = os.path.join(DATA_DIR, 'users.txt')
MOVIES_FILE = os.path.join(DATA_DIR, 'movie_titles.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'netflix_train.txt')
TEST_FILE = os.path.join(DATA_DIR, 'netflix_test.txt')

N_MOVIES = 10000  
K_CF = 100         # Increased for better performance
MF_K = 100           # Increase latent dimensions
MF_LAMBDA = 0.01    # Reduce regularization
MF_LR = 0.005       # Increase learning rate slightly
MF_EPOCHS = 20      # More epochs for convergence
MF_BATCH_SIZE = 50000  # Larger batch size
SAMPLE_FOR_DEBUG = False  
DEBUG_NUM_USERS = 1000  # Reduced for faster debugging
DEBUG_NUM_MOVIES = 1000

# -----------------------------
# Optimized User-Based Recommender Class
# -----------------------------
class UserBasedRecommender:
    def __init__(self, R_csr, user_means=None, k=50, min_common=5):
        self.R_csr = R_csr
        self.user_means = user_means
        self.k = k
        self.min_common = min_common
        self.user_similarities = {}  # Cache for similarities
        self.global_mean = 3.0
    
    def compute_user_similarity(self, u, v):
        """Compute similarity between two users - O(min(|u|, |v|))"""
        if u == v:
            return 1.0
            
        if (u, v) in self.user_similarities:
            return self.user_similarities[(u, v)]
        
        # Get user ratings in sparse format
        u_start, u_end = self.R_csr.indptr[u], self.R_csr.indptr[u+1]
        v_start, v_end = self.R_csr.indptr[v], self.R_csr.indptr[v+1]
        
        if (u_end - u_start) < self.min_common or (v_end - v_start) < self.min_common:
            self.user_similarities[(u, v)] = 0.0
            return 0.0
        
        u_items = self.R_csr.indices[u_start:u_end]
        u_ratings = self.R_csr.data[u_start:u_end]
        v_items = self.R_csr.indices[v_start:v_end] 
        v_ratings = self.R_csr.data[v_start:v_end]
        
        # Find common items efficiently using sorted intersection
        common_mask_u = np.isin(u_items, v_items, assume_unique=True)
        common_count = np.sum(common_mask_u)
        
        if common_count < self.min_common:
            self.user_similarities[(u, v)] = 0.0
            return 0.0
        
        u_common = u_ratings[common_mask_u]
        v_common_mask = np.isin(v_items, u_items, assume_unique=True)
        v_common = v_ratings[v_common_mask]
        
        # Mean centering if user means are available
        if self.user_means is not None:
            u_mean = self.user_means[u] if not np.isnan(self.user_means[u]) else self.global_mean
            v_mean = self.user_means[v] if not np.isnan(self.user_means[v]) else self.global_mean
            u_common = u_common - u_mean
            v_common = v_common - v_mean
        
        # Cosine similarity
        dot_product = np.dot(u_common, v_common)
        norm_u = np.linalg.norm(u_common)
        norm_v = np.linalg.norm(v_common)
        
        if norm_u > 1e-9 and norm_v > 1e-9:
            sim = dot_product / (norm_u * norm_v)
        else:
            sim = 0.0
        
        self.user_similarities[(u, v)] = sim
        self.user_similarities[(v, u)] = sim  # Symmetric
        return sim
    
    def predict(self, u, i, R_csc):
        """O(k) prediction after similarity precomputation"""
        user_mean = self.user_means[u] if self.user_means is not None and not np.isnan(self.user_means[u]) else self.global_mean
        
        # Get users who rated item i - O(degree(i))
        col_start, col_end = R_csc.indptr[i], R_csc.indptr[i+1]
        if col_end - col_start == 0:
            return user_mean
        
        users_who_rated = R_csc.indices[col_start:col_end]
        ratings_on_i = R_csc.data[col_start:col_end]
        
        # Remove target user if present
        mask = users_who_rated != u
        users_who_rated = users_who_rated[mask]
        ratings_on_i = ratings_on_i[mask]
        
        if len(users_who_rated) == 0:
            return user_mean
        
        # Collect similarities - O(k × avg_ratings_per_user)
        similarities = []
        valid_ratings = []
        
        for idx, (v, rating) in enumerate(zip(users_who_rated, ratings_on_i)):
            sim = self.compute_user_similarity(u, v)
            if sim > 0:
                similarities.append(sim)
                valid_ratings.append(rating)
                
            # Early stopping if we have enough candidates
            if len(similarities) >= self.k * 3:  
                break
        
        if not similarities:
            return user_mean
        
        # O(k) - Select top-k and predict
        similarities = np.array(similarities)
        valid_ratings = np.array(valid_ratings)
        
        top_k = min(self.k, len(similarities))
        if top_k == 0:
            return user_mean
            
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        
        top_sims = similarities[top_indices]
        top_ratings = valid_ratings[top_indices]
        
        # Filter out any zero or negative similarities (shouldn't happen but safe)
        positive_mask = top_sims > 0
        if not np.any(positive_mask):
            return user_mean
            
        top_sims = top_sims[positive_mask]
        top_ratings = top_ratings[positive_mask]
        
        prediction = np.dot(top_sims, top_ratings) / np.sum(top_sims)
        return np.clip(prediction, 1.0, 5.0)

    def batch_predict(self, test_triplets, R_csc, sample_limit=None):
        """Batch prediction for evaluation"""
        preds = []
        truths = []
        
        test_subset = test_triplets[:sample_limit] if sample_limit else test_triplets
        
        for (u, v, r) in tqdm(test_subset, desc="User-Based CF predictions"):
            try:
                p = self.predict(u, v, R_csc)
                preds.append(p)
                truths.append(r)
            except Exception as e:
                preds.append(self.global_mean)
                truths.append(r)
                continue
                
        return preds, truths

# -----------------------------
# Utilities
# -----------------------------
def rmse(preds, truths):
    """Calculate RMSE safely"""
    if len(preds) == 0:
        return float('inf')
    preds = np.array(preds, dtype=np.float64)
    truths = np.array(truths, dtype=np.float64)
    mask = ~(np.isnan(preds) | np.isinf(preds))
    if np.sum(mask) == 0:
        return float('inf')
    return float(np.sqrt(np.mean((preds[mask] - truths[mask]) ** 2)))

# -----------------------------
# Data loading / preprocessing - DEBUGGED
# -----------------------------
def load_user_list(users_path):
    """Load user list with error handling"""
    users = []
    try:
        with open(users_path, 'r') as f:
            for ln in f:
                s = ln.strip()
                if s:
                    users.append(int(s))
    except FileNotFoundError:
        print(f"Error: {users_path} not found")
        sys.exit(1)
    uid_to_idx = {uid: idx for idx, uid in enumerate(users)}
    return users, uid_to_idx

def load_movie_titles(movie_titles_path):
    """Load movie titles with error handling"""
    titles = {}
    try:
        with open(movie_titles_path, 'r', encoding='latin1') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    parts = ln.split(',', 2)
                    if len(parts) >= 3:
                        try:
                            mid = int(parts[0])
                            titles[mid] = (parts[1], parts[2])
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Warning: {movie_titles_path} not found")
    return titles

def load_train_triplets(train_path, uid_to_idx, n_movies, sample_mode=False, debug_users=None, debug_movies=None):
    """Load training data with proper sampling"""
    rows, cols, vals = [], [], []
    triplets = []
    
    if sample_mode:
        print(f"Sampling mode: using {debug_users} users and {debug_movies} movies")
    
    try:
        with open(train_path, 'r') as f:
            for ln in tqdm(f, desc="Loading training data"):
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 3:
                    continue
                    
                uid = int(parts[0]); mid = int(parts[1]); rating = float(parts[2])
                
                if uid not in uid_to_idx:
                    continue
                    
                u = uid_to_idx[uid]
                v = mid - 1  # Convert to 0-based indexing
                
                if sample_mode:
                    if u >= debug_users or v >= debug_movies:
                        continue
                
                rows.append(u); cols.append(v); vals.append(rating)
                triplets.append((u, v, rating))
    except FileNotFoundError:
        print(f"Error: {train_path} not found")
        sys.exit(1)
    
    if sample_mode:
        m, n = debug_users, debug_movies
    else:
        m = len(uid_to_idx)
        n = n_movies
        
    R = sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float32)
    return R, triplets

def load_test_triplets(test_path, uid_to_idx, sample_mode=False, debug_users=None, debug_movies=None):
    """Load test data with proper sampling"""
    triplets = []
    
    try:
        with open(test_path, 'r') as f:
            for ln in tqdm(f, desc="Loading test data"):
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 3:
                    continue
                    
                uid = int(parts[0]); mid = int(parts[1]); rating = float(parts[2])
                
                if uid not in uid_to_idx:
                    continue
                    
                u = uid_to_idx[uid]; v = mid - 1
                
                if sample_mode:
                    if u >= debug_users or v >= debug_movies:
                        continue
                        
                triplets.append((u, v, rating))
    except FileNotFoundError:
        print(f"Error: {test_path} not found")
        sys.exit(1)
        
    return triplets

# -----------------------------
# Baselines
# -----------------------------
def compute_baselines(R_csr, train_triplets):
    """Compute baseline statistics safely"""
    if len(train_triplets) == 0:
        return 3.0, np.array([3.0]), np.array([3.0])
    
    data = np.array([t[2] for t in train_triplets], dtype=np.float64)
    global_mean = float(np.mean(data))
    
    # User means
    sums = np.array(R_csr.sum(axis=1)).reshape(-1)
    counts = np.diff(R_csr.indptr)
    user_means = np.full_like(sums, global_mean, dtype=np.float64)
    valid_users = counts > 0
    user_means[valid_users] = sums[valid_users] / counts[valid_users]
    
    # Item means  
    R_csc = R_csr.tocsc()
    sums_i = np.array(R_csc.sum(axis=0)).reshape(-1)
    counts_i = np.diff(R_csc.indptr)
    item_means = np.full_like(sums_i, global_mean, dtype=np.float64)
    valid_items = counts_i > 0
    item_means[valid_items] = sums_i[valid_items] / counts_i[valid_items]
    
    return global_mean, user_means, item_means

# -----------------------------
# Matrix factorization (SGD) - DEBUGGED
# -----------------------------
def sgd_matrix_factorization_simple(train_triplets, m, n, k=20, lr=0.001, reg=0.1, epochs=15, eval_triplets=None):
    """Simplified and more stable matrix factorization"""
    if len(train_triplets) == 0:
        return None, None, {'train_rmse': [], 'test_rmse': [], 'obj': []}
    
    rng = np.random.RandomState(42)
    U = 0.01 * rng.randn(m, k).astype(np.float32)
    V = 0.01 * rng.randn(n, k).astype(np.float32)
    
    # Convert to arrays for faster access
    triplets_u = np.array([t[0] for t in train_triplets], dtype=np.int32)
    triplets_v = np.array([t[1] for t in train_triplets], dtype=np.int32) 
    triplets_r = np.array([t[2] for t in train_triplets], dtype=np.float32)
    
    history = {'train_rmse': [], 'test_rmse': [], 'obj': []}
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # Shuffle training data
        indices = np.random.permutation(len(train_triplets))
        
        # SGD updates
        for idx in indices:
            u = triplets_u[idx]
            v = triplets_v[idx]
            r = triplets_r[idx]
            
            # Prediction and error
            pred = np.dot(U[u], V[v])
            err = pred - r
            
            # Gradient updates
            U[u] -= lr * (err * V[v] + 2 * reg * U[u])
            V[v] -= lr * (err * U[u] + 2 * reg * V[v])
        
        # Evaluation
        train_preds = np.array([np.dot(U[triplets_u[i]], V[triplets_v[i]]) for i in range(min(10000, len(train_triplets)))])
        train_truths = triplets_r[:len(train_preds)]
        train_rmse = rmse(train_preds, train_truths)
        history['train_rmse'].append(train_rmse)
        
        if eval_triplets is not None:
            #eval_sample = eval_triplets[:5000]  # Sample for speed
            test_preds = [np.dot(U[u], V[v]) for (u, v, _) in eval_triplets]
            test_truths = [r for (_, _, r) in eval_triplets]
            test_rmse = rmse(test_preds, test_truths)
            history['test_rmse'].append(test_rmse)
        else:
            test_rmse = None
            
        # Objective (approximate)
        obj = 0.5 * np.mean((train_preds - train_truths) ** 2) + reg * (np.mean(U ** 2) + np.mean(V ** 2))
        history['obj'].append(obj)
        
        epoch_time = time.time() - t0
        print(f"[MF] Epoch {epoch}/{epochs} | Time: {epoch_time:.1f}s | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse if test_rmse else 'N/A':.4f}")
    
    return U, V, history

# -----------------------------
# Main pipeline - DEBUGGED
# -----------------------------
def main():
    print("=== Netflix Recommendation System ===\n")
    
    # Check if data files exist
    required_files = [USERS_FILE, TRAIN_FILE, TEST_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file {f} not found!")
            print("Please ensure all data files are in the current directory.")
            sys.exit(1)
    
    print("Loading user list...")
    users, uid_to_idx = load_user_list(USERS_FILE)
    m = len(users)
    print(f"Users loaded: {m}")
    
    print("Loading movie titles...")
    movie_titles = load_movie_titles(MOVIES_FILE)
    print(f"Movies loaded: {len(movie_titles)}")
    
    print("Loading training data...")
    if SAMPLE_FOR_DEBUG:
        R_train, train_triplets = load_train_triplets(TRAIN_FILE, uid_to_idx, N_MOVIES, 
                                                    sample_mode=True, 
                                                    debug_users=DEBUG_NUM_USERS, 
                                                    debug_movies=DEBUG_NUM_MOVIES)
    else:
        R_train, train_triplets = load_train_triplets(TRAIN_FILE, uid_to_idx, N_MOVIES)
    
    print(f"Training data: {len(train_triplets)} ratings")
    print(f"Matrix shape: {R_train.shape}, Density: {R_train.nnz / (R_train.shape[0] * R_train.shape[1]):.4f}")
    
    print("Loading test data...")
    if SAMPLE_FOR_DEBUG:
        test_triplets = load_test_triplets(TEST_FILE, uid_to_idx, 
                                         sample_mode=True,
                                         debug_users=DEBUG_NUM_USERS,
                                         debug_movies=DEBUG_NUM_MOVIES)
    else:
        test_triplets = load_test_triplets(TEST_FILE, uid_to_idx)
    
    print(f"Test data: {len(test_triplets)} ratings")
    
    # Convert to CSC format
    R_csc = R_train.tocsc()
    
    # Compute baselines
    print("\nComputing baselines...")
    global_mean, user_means, item_means = compute_baselines(R_train, train_triplets)
    print(f"Global mean rating: {global_mean:.4f}")
    
    # Baseline predictions
    test_truths = [r for (_, _, r) in test_triplets]
    
    # Global mean baseline
    global_preds = [global_mean] * len(test_triplets)
    global_rmse = rmse(global_preds, test_truths)
    print(f"Global mean baseline RMSE: {global_rmse:.4f}")
    
    # User mean baseline
    user_mean_preds = [user_means[u] if not np.isnan(user_means[u]) else global_mean for (u, _, _) in test_triplets]
    user_mean_rmse = rmse(user_mean_preds, test_truths)
    print(f"User mean baseline RMSE: {user_mean_rmse:.4f}")
    
    # Item mean baseline  
    item_mean_preds = [item_means[v] if not np.isnan(item_means[v]) else global_mean for (_, v, _) in test_triplets]
    item_mean_rmse = rmse(item_mean_preds, test_truths)
    print(f"Item mean baseline RMSE: {item_mean_rmse:.4f}")
    
    # Optimized User-based Collaborative Filtering
    print(f"\nRunning User-Based CF (k={K_CF})...")
    cf_recommender = UserBasedRecommender(
        R_train, 
        user_means=user_means, 
        k=K_CF,
        min_common=5
    )
    
    # Warm up the similarity cache with some common users
    print("Warming up similarity cache...")
    warm_up_users = min(100, R_train.shape[0])
    for i in tqdm(range(warm_up_users)):
        for j in range(i+1, min(i+10, warm_up_users)):
            cf_recommender.compute_user_similarity(i, j)
    
    # Run predictions
    cf_preds, cf_truths = cf_recommender.batch_predict(test_triplets, R_csc, sample_limit=None)
    cf_rmse = rmse(cf_preds, cf_truths)
    print(f"User-Based CF RMSE (on {len(cf_preds)} samples): {cf_rmse:.4f}")
    
    # Matrix Factorization
    print(f"\nRunning Matrix Factorization (k={MF_K}, λ={MF_LAMBDA})...")
    U, V, mf_history = sgd_matrix_factorization_simple(
        train_triplets, 
        R_train.shape[0], 
        R_train.shape[1],
        k=MF_K,
        lr=MF_LR,
        reg=MF_LAMBDA,
        epochs=MF_EPOCHS,
        eval_triplets=test_triplets
    )
    
    # Final MF evaluation
    if U is not None and V is not None:
        mf_preds = []
        for (u, v, r) in test_triplets:  
            try:
                pred = np.dot(U[u], V[v])
                mf_preds.append(np.clip(pred, 1.0, 5.0))
            except:
                mf_preds.append(global_mean)
        
        mf_truths = [r for (_, _, r) in test_triplets]
        final_mf_rmse = rmse(mf_preds, mf_truths)
        print(f"Final Matrix Factorization RMSE: {final_mf_rmse:.4f}")
    else:
        final_mf_rmse = float('inf')
    
    # Plot results
    print("\nGenerating plots...")
    if mf_history['train_rmse']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(mf_history['obj'], 'b-', linewidth=2)
        plt.title('MF Training Objective')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(mf_history['train_rmse'], 'g-', label='Train RMSE', linewidth=2)
        if mf_history['test_rmse']:
            plt.plot(mf_history['test_rmse'], 'r-', label='Test RMSE', linewidth=2)
        plt.title('MF Training RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Saved training curves to 'training_curves.png'")
    
    # Summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"Global Mean Baseline: {global_rmse:.4f}")
    print(f"User Mean Baseline: {user_mean_rmse:.4f}")
    print(f"Item Mean Baseline: {item_mean_rmse:.4f}")
    print(f"User-Based CF: {cf_rmse:.4f}")
    print(f"Matrix Factorization: {final_mf_rmse:.4f}")
    
    best_method = "Matrix Factorization" if final_mf_rmse < cf_rmse else "User-Based CF"
    print(f"\nBest method: {best_method}")
    print("Experiment completed successfully!")

if __name__ == '__main__':
    main()