import numpy as np
import scipy.sparse
from .estimator_v1_2_0 import HybridHistGBMNumbaV1_2_0
from .sparse_engine import (
    bin_sparse_features_hybrid_numba,
    find_zero_bins,
    build_histogram_sparse_njit,
    partition_indices_sparse_njit,
    predict_all_trees_sparse_jit
)

class HybridHistGBMSparse(HybridHistGBMNumbaV1_2_0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sparse = False

    def fit(self, X, y, eval_set=None):
        if scipy.sparse.issparse(X):
            self.is_sparse = True
            if not scipy.sparse.isspmatrix_csr(X):
                X = X.tocsr()
            self._n_samples, self.n_features_in_ = X.shape
            
            # Setup features and categorical
            if self.categorical_features == 'auto':
                self.categorical_indices = []
            elif self.categorical_features:
                self.categorical_indices = [int(c) for c in self.categorical_features]
            else:
                self.categorical_indices = []
                
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            
            # Use original logic for edges (requires some dense sample for quantiles if purely sparse)
            # We take a small dense sample to build bin edges
            dense_sample_size = min(100000, self._n_samples)
            X_sample = X[:dense_sample_size].toarray()
            self.bin_edges_, self.bin_counts_ = self._build_bin_edges(X_sample)
            
            # Bin the sparse data
            self.X_bin_data = bin_sparse_features_hybrid_numba(
                X.data, X.indices, X.indptr, self.bin_edges_, self.bin_counts_
            )
            self.zero_bins = find_zero_bins(self.bin_edges_, self.bin_counts_)
            self.X_indices = X.indices
            self.X_indptr = X.indptr
            
            # Base logic initialization
            if self.n_classes_ > 2:
                self.objective = 'multiclass'
                y_encoded = np.zeros_like(y)
                for i, c in enumerate(self.classes_):
                    y_encoded[y == c] = i
                y = y_encoded
                self.base_score_ = np.ones((self._n_samples, self.n_classes_)) / self.n_classes_
                self.trees_ = [[] for _ in range(self.n_classes_)]
                
                # Start tree building loop (simplified for demonstration)
                # In production, we'd replicate the full loop. For the sake of this architectural test,
                # we will override the core _build_tree_leafwise call.
            else:
                self.objective = 'binary'
                
            # We will use the parent fit, but we monkeypatch the dense X_bin 
            # Actually, calling super().fit(X) will fail because it expects dense.
            print("Sparse Engine Initialized. Overriding core build paths.")
            self._fit_sparse(X, y, eval_set)
            return self
        else:
            return super().fit(X, y, eval_set)

    def _fit_sparse(self, X, y, eval_set):
        # A simplified training loop purely to demonstrate the sparse math engine works
        # This builds exactly 1 tree or multiple trees without early stopping for the test
        if self.objective == 'binary':
            F = np.full(self._n_samples, 0.0)
            self.trees_ = []
            
            for i in range(self.n_estimators):
                probs = 1.0 / (1.0 + np.exp(-F))
                grad = probs - y
                hess = probs * (1.0 - probs)
                
                # Subsampling
                sample_idx = np.arange(self._n_samples)
                feat_idx = np.arange(self.n_features_in_)
                
                tree = self._build_tree_sparse(grad, hess, sample_idx, feat_idx)
                self.trees_.append(tree)
                
                # Predict to update F
                preds = predict_all_trees_sparse_jit(
                    self.X_bin_data, self.X_indices, self.X_indptr, self.zero_bins,
                    np.array([tree['features']]),
                    np.array([tree['thresholds']]),
                    np.array([tree['values']]),
                    np.array([tree['left']]),
                    np.array([tree['right']]),
                    1, self.learning_rate
                )
                F += preds
                
        else:
            # Multiclass
            print("Multiclass Sparse not fully mapped in this prototype. Run binary test.")
            
    def _build_tree_sparse(self, grad, hess, sample_indices, feature_indices):
        from .sparse_engine import build_histogram_sparse_njit, partition_indices_sparse_njit
        
        # Simplified tree builder for demonstration
        max_nodes = (1 << (self.max_depth + 1)) - 1
        features = np.full(max_nodes, -1, dtype=np.int32)
        thresholds = np.full(max_nodes, 0, dtype=np.int32)
        values = np.zeros(max_nodes, dtype=np.float64)
        left = np.full(max_nodes, -1, dtype=np.int32)
        right = np.full(max_nodes, -1, dtype=np.int32)
        
        nodes = [(0, 0, self._n_samples, 0)] # (node_id, start, end, depth)
        
        while nodes:
            node_id, start, end, depth = nodes.pop(0)
            
            hist_G, hist_H, total_G, total_H = build_histogram_sparse_njit(
                self.X_bin_data, self.X_indices, self.X_indptr, self.zero_bins,
                grad, hess, sample_indices[start:end], feature_indices, self._max_bins
            )
            
            # Calculate weight
            weight = -total_G / (total_H + self.reg_lambda) if (total_H + self.reg_lambda) > 0 else 0.0
            values[node_id] = weight
            
            if depth >= self.max_depth or (end - start) < self.min_samples_leaf:
                continue
                
            # Find best split (naive python loop for prototype, usually JIT)
            best_gain = -1e9
            best_feat = -1
            best_bin = -1
            
            root_score = (total_G**2) / (total_H + self.reg_lambda)
            
            for f_idx in range(len(feature_indices)):
                sum_g_L = 0.0
                sum_h_L = 0.0
                for b in range(self._max_bins - 1):
                    sum_g_L += hist_G[f_idx, b]
                    sum_h_L += hist_H[f_idx, b]
                    sum_g_R = total_G - sum_g_L
                    sum_h_R = total_H - sum_h_L
                    
                    if sum_h_L < self.min_child_weight or sum_h_R < self.min_child_weight:
                        continue
                        
                    gain = (sum_g_L**2)/(sum_h_L + self.reg_lambda) + (sum_g_R**2)/(sum_h_R + self.reg_lambda) - root_score
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feature_indices[f_idx]
                        best_bin = b
                        
            if best_feat != -1 and best_gain > self.gamma:
                features[node_id] = best_feat
                thresholds[node_id] = best_bin
                left[node_id] = node_id * 2 + 1
                right[node_id] = node_id * 2 + 2
                
                mid = partition_indices_sparse_njit(
                    self.X_bin_data, self.X_indices, self.X_indptr, self.zero_bins,
                    sample_indices, best_feat, best_bin, start, end
                )
                nodes.append((left[node_id], start, mid, depth + 1))
                nodes.append((right[node_id], mid, end, depth + 1))
                
        return {
            'features': features,
            'thresholds': thresholds,
            'values': values,
            'left': left,
            'right': right
        }

    def predict(self, X):
        if scipy.sparse.issparse(X):
            if not scipy.sparse.isspmatrix_csr(X):
                X = X.tocsr()
            
            X_bin_data = bin_sparse_features_hybrid_numba(
                X.data, X.indices, X.indptr, self.bin_edges_, self.bin_counts_
            )
            
            all_f = np.array([t['features'] for t in self.trees_])
            all_t = np.array([t['thresholds'] for t in self.trees_])
            all_v = np.array([t['values'] for t in self.trees_])
            all_l = np.array([t['left'] for t in self.trees_])
            all_r = np.array([t['right'] for t in self.trees_])
            
            preds = predict_all_trees_sparse_jit(
                X_bin_data, X.indices, X.indptr, self.zero_bins,
                all_f, all_t, all_v, all_l, all_r,
                self.n_estimators, self.learning_rate
            )
            return (preds > 0).astype(int)
        else:
            return super().predict(X)
