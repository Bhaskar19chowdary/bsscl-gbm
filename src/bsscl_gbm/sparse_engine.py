"""
Sparse Matrix Engine for BSSCL-GBM
Handles scipy.sparse.csr_matrix natively in Numba JIT
"""
import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def bin_sparse_features_hybrid_numba(X_data, X_indices, X_indptr, edges_2d, bin_counts):
        """
        Bins only the non-zero elements of a CSR matrix.
        Returns X_bin_data of the exact same length as X_data.
        """
        n_elements = len(X_data)
        X_bin_data = np.empty(n_elements, dtype=np.uint8)
        
        for i in prange(n_elements):
            v = X_data[i]
            j = X_indices[i]
            
            nb = bin_counts[j]
            last_idx = nb - 1
            b_first = edges_2d[j, 0]
            b_end = edges_2d[j, nb]
            scale = last_idx / (b_end - b_first) if b_end > b_first else 1.0
            
            if v < b_first:
                X_bin_data[i] = 0
            elif v >= b_end:
                X_bin_data[i] = last_idx
            else:
                g = int((v - b_first) * scale)
                if g < 0: g = 0
                elif g > last_idx: g = last_idx
                
                if edges_2d[j, g] <= v and (g == last_idx or v < edges_2d[j, g + 1]):
                    X_bin_data[i] = g
                else:
                    lo = g - 4
                    hi = g + 4
                    lo = max(lo, 0)
                    hi = min(hi, last_idx)
                    
                    if v >= edges_2d[j, lo] and (hi == last_idx or v < edges_2d[j, hi + 1]):
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if edges_2d[j, mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        X_bin_data[i] = lo
                    else:
                        lo, hi = 0, last_idx
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if edges_2d[j, mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        X_bin_data[i] = lo
        return X_bin_data

    @njit(cache=True)
    def find_zero_bins(edges_2d, bin_counts):
        """Finds which bin '0.0' falls into for every feature."""
        p = len(bin_counts)
        zero_bins = np.zeros(p, dtype=np.uint8)
        for j in range(p):
            nb = bin_counts[j]
            b_first = edges_2d[j, 0]
            b_end = edges_2d[j, nb]
            if 0.0 < b_first:
                zero_bins[j] = 0
            elif 0.0 >= b_end:
                zero_bins[j] = nb - 1
            else:
                lo, hi = 0, nb - 1
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if edges_2d[j, mid + 1] <= 0.0:
                        lo = mid + 1
                    else:
                        hi = mid
                zero_bins[j] = lo
        return zero_bins

    @njit(parallel=True, cache=True)
    def build_histogram_sparse_njit(X_bin_data, X_indices, X_indptr, zero_bins, grad, hess, sample_indices, feature_indices,
                                    max_bins):
        """
        Builds histograms using sparse traversal.
        Calculates implicit zero-bin gradients mathematically.
        """
        n_samples = len(sample_indices)
        n_selected_features = len(feature_indices)
        
        # We need a reverse mapping to know if a column is selected and what its local index is
        # We assume maximum features < 1000000 for mapping array
        max_feat = 0
        if n_selected_features > 0:
            for f in feature_indices:
                if f > max_feat:
                    max_feat = f
        
        # Map: global feature index -> local selected index (-1 if not selected)
        feat_map = np.full(max_feat + 1, -1, dtype=np.int32)
        for j in range(n_selected_features):
            feat_map[feature_indices[j]] = j
            
        hist_G = np.zeros((n_selected_features, max_bins), dtype=np.float64)
        hist_H = np.zeros((n_selected_features, max_bins), dtype=np.float64)
        
        # Total sums
        total_G = 0.0
        total_H = 0.0
        for i in prange(n_samples):
            row = sample_indices[i]
            total_G += grad[row]
            total_H += hess[row]

        # Parallel over samples? Race conditions on hist_G!
        # Must do thread-local histograms or parallel over features.
        # For simplicity, we parallelize over features!
        # Wait, iterating CSR matrix by feature is slow (requires CSC).
        # We will do sequential sample iteration, but parallel feature correction.
        # Actually, let's just make it parallel over threads with thread-local storage if possible.
        # Since numba doesn't easily expose thread ID, we'll just do sequential for the histogram loop, 
        # or use Numba's fast atomic adds if needed. For now, sequential iteration over sample_indices:
        
        for i in range(n_samples):
            row = sample_indices[i]
            g = grad[row]
            h = hess[row]
            start = X_indptr[row]
            end = X_indptr[row+1]
            for ptr in range(start, end):
                col = X_indices[ptr]
                if col <= max_feat:
                    local_j = feat_map[col]
                    if local_j != -1:
                        b = X_bin_data[ptr]
                        hist_G[local_j, b] += g
                        hist_H[local_j, b] += h

        # Now correct the implicit zero bins
        for j in prange(n_selected_features):
            orig_j = feature_indices[j]
            z_bin = zero_bins[orig_j]
            
            sum_g = 0.0
            sum_h = 0.0
            for b in range(max_bins):
                sum_g += hist_G[j, b]
                sum_h += hist_H[j, b]
                
            missing_g = total_G - sum_g
            missing_h = total_H - sum_h
            
            hist_G[j, z_bin] += missing_g
            hist_H[j, z_bin] += missing_h
            
        return hist_G, hist_H, total_G, total_H

    @njit(cache=True)
    def partition_indices_sparse_njit(X_bin_data, X_indices, X_indptr, zero_bins, sample_indices, best_feature, best_bin, start, end):
        """
        Partitions indices based on a split condition in a sparse matrix.
        """
        left = np.empty(end - start, dtype=np.int32)
        right = np.empty(end - start, dtype=np.int32)
        l_cnt = 0
        r_cnt = 0
        
        z_bin = zero_bins[best_feature]
        
        for i in range(start, end):
            row = sample_indices[i]
            ptr_start = X_indptr[row]
            ptr_end = X_indptr[row+1]
            
            # Find feature in row
            bin_val = z_bin
            # Linear scan since nnz per row is usually small
            for ptr in range(ptr_start, ptr_end):
                if X_indices[ptr] == best_feature:
                    bin_val = X_bin_data[ptr]
                    break
            
            if bin_val <= best_bin:
                left[l_cnt] = row
                l_cnt += 1
            else:
                right[r_cnt] = row
                r_cnt += 1
                
        # Write back
        for i in range(l_cnt):
            sample_indices[start + i] = left[i]
        for i in range(r_cnt):
            sample_indices[start + l_cnt + i] = right[i]
            
        return start + l_cnt

    @njit(parallel=True, cache=True)
    def predict_all_trees_sparse_jit(X_bin_data, X_indices, X_indptr, zero_bins, all_features, all_thresholds, all_values,
                                     left_children, right_children, n_trees, learning_rate):
        n_samples = len(X_indptr) - 1
        preds = np.zeros(n_samples, dtype=np.float64)
        
        for i in prange(n_samples):
            ptr_start = X_indptr[i]
            ptr_end = X_indptr[i+1]
            
            # Predict over all trees
            for t in range(n_trees):
                node = 0
                while True:
                    f = all_features[t, node]
                    if f == -1: # Leaf
                        preds[i] += learning_rate * all_values[t, node]
                        break
                    
                    # Find feature f in row
                    bin_val = zero_bins[f]
                    for ptr in range(ptr_start, ptr_end):
                        if X_indices[ptr] == f:
                            bin_val = X_bin_data[ptr]
                            break
                    
                    if bin_val <= all_thresholds[t, node]:
                        node = left_children[t, node]
                    else:
                        node = right_children[t, node]
                        
        return preds
