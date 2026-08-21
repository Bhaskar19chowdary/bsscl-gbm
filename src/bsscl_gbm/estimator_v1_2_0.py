"""
Hybrid + Numba JIT Gradient Boosting Machine (v3.0 — Enterprise Edition)
========================================================================

NEW IN v3.0 (over v2.8):
  ✓ Early Stopping with Validation Tracking — auto-stops when validation plateaus
  ✓ Monotonic Constraints — enforce directional rules on features
  ✓ Sparsity-Aware Split Finding — native NaN/missing value handling
  ✓ Native Categorical Feature Support — no One-Hot Encoding required

NEW IN v2.8 (over v2.7):
  ✓ Production Deployment Support (Multiprocessing-Safe)
  ✓ warmup() method — pre-compile JIT at container startup
  ✓ predict_batch_parallel() — safe multi-core batch prediction
  ✓ predict_proba_batch_parallel() — safe multi-core probability prediction
  ✓ production_info() — deployment config and recommendations

NEW IN v2.7 (over v2.6):
  ✓ Leaf-Wise (Best-First) tree growth  — like LightGBM
  ✓ max_leaves parameter (default=31) replaces max_depth as primary control
  ✓ Vectorized gradient/hessian computation (numpy matrix ops)
  ✓ Vectorized one-hot encoding
  ✓ grow_policy param: 'leaf' (fast, default) or 'level' (original)

NEW IN v2.6 (over v2.5):
  ✓ Multi-class classification via softmax (K trees per round)
  ✓ Auto-detection: binary (sigmoid) vs multi-class (softmax)
  ✓ Label remapping for arbitrary class labels
  ✓ Balanced class weights for K classes

CARRIED FROM v2.5:
  ✓ Stochastic row subsampling (subsample=0.8)
  ✓ Column subsampling per tree (colsample_bytree=0.8)
  ✓ Histogram subtraction trick
  ✓ min_child_weight and gamma regularization
  ✓ Tuned adaptive hyperparameters

Author: Bhaskar
Version: 3.0 (Enterprise Edition)
Date: July 2026
"""

import concurrent.futures
import time
import os
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA JIT
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from numba import njit, prange, cuda
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠️  Numba not installed. Using pure Python (slower)")

if HAS_NUMBA:

    # ─── Binning ───

    @njit(cache=True)
    def standard_binning_search(arr, target):
        low, high = 0, len(arr) - 2
        if target < arr[0]: return 0
        if target >= arr[-1]: return len(arr) - 2
        while low <= high:
            mid = (low + high) >> 1
            if arr[mid] <= target < arr[mid + 1]: return mid
            elif target < arr[mid]: high = mid - 1
            else: low = mid + 1
        return max(min(low, len(arr) - 2), 0)

    @njit(cache=True)
    def advanced_binning_search(arr, target):
        low, high = 0, len(arr) - 1
        if target < arr[0]: return 0
        if target >= arr[high]: return high - 1
        while low + 1 <= high - 1:
            mid = ((low + 1) + (high - 1)) >> 1
            if arr[low] <= target < arr[low + 1]: return low
            if arr[high - 1] <= target < arr[high]: return high - 1
            if arr[mid] <= target < arr[mid + 1]: return mid
            if target < arr[mid]: high = mid - 1
            else: low = mid + 1
        return max(high - 1, 0)

    @njit(parallel=True, cache=True)
    def hybrid_binning_numba(bins, values):
        n = len(values)
        n_bins = len(bins)
        indices = np.empty(n, dtype=np.int32)
        
        b_first = bins[0]
        last_idx = n_bins - 2
        b_end = bins[-1]
        scale = last_idx / (b_end - b_first) if b_end > b_first else 1.0
        
        for i in prange(n):
            v = values[i]
            # O(1) edge snapping
            if v < b_first:
                indices[i] = 0
            elif v >= b_end:
                indices[i] = last_idx
            else:
                # Interpolation guess — single multiply
                g = int((v - b_first) * scale)
                if g < 0:
                    g = 0
                elif g > last_idx:
                    g = last_idx
                
                # Fast path: guess is correct (most common for quantile bins)
                if bins[g] <= v and (g == last_idx or v < bins[g + 1]):
                    indices[i] = g
                else:
                    # Bounded window around the guess
                    lo = g - 4
                    hi = g + 4
                    lo = max(lo, 0)
                    hi = min(hi, last_idx)
                    
                    # Check if value is within the bounded window
                    if v >= bins[lo] and (hi == last_idx or v < bins[hi + 1]):
                        # Tight binary search in [lo, hi]
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if bins[mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        indices[i] = lo
                    else:
                        # Fallback: full binary search (rare — skewed data)
                        lo = 0
                        hi = last_idx
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if bins[mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        indices[i] = lo
        return indices

    @njit(parallel=True, cache=True)
    def standard_only_binning_numba(bins, values):
        n = len(values)
        indices = np.empty(n, dtype=np.int32)
        for i in prange(n):
            indices[i] = standard_binning_search(bins, values[i])
        return indices
    @njit(parallel=True, cache=True)
    def bin_all_features_hybrid_numba(X, edges_2d, bin_counts, X_binned):
        n, p = X.shape
        
        for i in prange(n):
            for j in range(p):
                v = X[i, j]
                nb = bin_counts[j]
                last_idx = nb - 1
                b_first = edges_2d[j, 0]
                b_end = edges_2d[j, nb]
                scale = last_idx / (b_end - b_first) if b_end > b_first else 1.0
                
                if v < b_first:
                    X_binned[i, j] = 0
                elif v >= b_end:
                    X_binned[i, j] = last_idx
                else:
                    g = int((v - b_first) * scale)
                    if g < 0: g = 0
                    elif g > last_idx: g = last_idx
                    
                    if edges_2d[j, g] <= v and (g == last_idx or v < edges_2d[j, g + 1]):
                        X_binned[i, j] = g
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
                            X_binned[i, j] = lo
                        else:
                            lo, hi = 0, last_idx
                            while lo < hi:
                                mid = (lo + hi) >> 1
                                if edges_2d[j, mid + 1] <= v:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            X_binned[i, j] = lo
        return X_binned

    @njit(parallel=True, cache=True)
    def bin_all_features_standard_numba(X, edges_2d, bin_counts, X_binned):
        n, p = X.shape
        for i in prange(n):
            for j in range(p):
                v = X[i, j]
                nb = bin_counts[j]
                lo = 0
                hi = nb - 1
                if v < edges_2d[j, 0]:
                    X_binned[i, j] = 0
                    continue
                if v >= edges_2d[j, nb]:
                    X_binned[i, j] = nb - 1
                    continue
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if edges_2d[j, mid + 1] <= v:
                        lo = mid + 1
                    else:
                        hi = mid
                X_binned[i, j] = lo
        return X_binned


    # ═══════════════════════════════════════════════════════════════════════
    # FULL JIT TREE BUILDING WITH HISTOGRAM SUBTRACTION
    # ═══════════════════════════════════════════════════════════════════════

    @njit(parallel=True, cache=True)
    def _build_tree_v25(X_bin, grad, hess, sample_indices, feature_indices,
                         max_depth, min_samples_leaf, min_child_weight,
                         bin_counts, max_bins, reg_lambda, gamma, n_all_features, is_categorical):
        """
        Build a tree with:
        1. Sample subsampling (only uses sample_indices rows)
        2. Feature subsampling (only evaluates feature_indices columns)
        3. Histogram subtraction trick (halves histogram work)
        4. min_child_weight + gamma regularization
        """
        n_samples = len(sample_indices)
        n_selected_features = len(feature_indices)
        max_nodes = (1 << (max_depth + 1)) - 1

        # Tree arrays
        tree_features = np.full(max_nodes, -1, dtype=np.int32)
        tree_thresholds = np.zeros(max_nodes, dtype=np.int64)
        tree_values = np.zeros(max_nodes, dtype=np.float64)
        tree_left = np.full(max_nodes, -1, dtype=np.int32)
        tree_right = np.full(max_nodes, -1, dtype=np.int32)
        split_gains = np.zeros(max_nodes, dtype=np.float64)
        node_count = np.int32(0)

        # Working indices (copy, will be partitioned in-place)
        indices = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            indices[i] = sample_indices[i]

        # Stack
        stack_size = max_depth * 2 + 4
        stack_node = np.empty(stack_size, dtype=np.int32)
        stack_start = np.empty(stack_size, dtype=np.int64)
        stack_end = np.empty(stack_size, dtype=np.int64)
        stack_depth = np.empty(stack_size, dtype=np.int32)
        stack_has_hist = np.zeros(stack_size, dtype=np.int32)
        stack_ptr = 0

        # Pre-allocated histogram storage per stack entry (for subtraction trick)
        # v4.0: Use only 2 slots (left/right) instead of stack_size to reduce RAM
        hist_G = np.zeros((stack_size, n_all_features, max_bins), dtype=np.float64)
        hist_H = np.zeros((stack_size, n_all_features, max_bins), dtype=np.float64)

        # Working histogram buffers
        G = np.zeros((n_all_features, max_bins), dtype=np.float64)
        H = np.zeros((n_all_features, max_bins), dtype=np.float64)

        # Push root
        root_id = node_count
        node_count += 1
        stack_node[stack_ptr] = root_id
        stack_start[stack_ptr] = 0
        stack_end[stack_ptr] = n_samples
        stack_depth[stack_ptr] = 0
        stack_has_hist[stack_ptr] = 0
        stack_ptr += 1

        while stack_ptr > 0:
            stack_ptr -= 1
            cur_node = stack_node[stack_ptr]
            start = stack_start[stack_ptr]
            end = stack_end[stack_ptr]
            depth = stack_depth[stack_ptr]
            has_hist = stack_has_hist[stack_ptr]
            cur_slot = stack_ptr  # slot we just popped from

            n_node = end - start

            # Compute leaf value
            g_sum = 0.0
            h_sum = 0.0
            for k in range(start, end):
                i = indices[k]
                g_sum += grad[i]
                h_sum += hess[i]
            leaf_val = -g_sum / (h_sum + reg_lambda)

            # Leaf conditions
            if depth >= max_depth or n_node < min_samples_leaf or h_sum < min_child_weight:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            # ─── Build or retrieve histogram ───
            if has_hist:
                # USE PRE-COMPUTED histogram (from subtraction trick)
                for jj in prange(n_selected_features):
                    j = feature_indices[jj]
                    nb = bin_counts[j]
                    for b in range(nb):
                        G[j, b] = hist_G[cur_slot, j, b]
                        H[j, b] = hist_H[cur_slot, j, b]
            else:
                # BUILD histogram from scratch (v4.0: Parallelized via prange)
                for jj in prange(n_selected_features):
                    j = feature_indices[jj]
                    nb = bin_counts[j]
                    for b in range(nb):
                        G[j, b] = 0.0
                        H[j, b] = 0.0
                    for k in range(start, end):
                        i = indices[k]
                        g = grad[i]
                        h = hess[i]
                        b = X_bin[i, j]
                        if b >= nb:
                            b = nb - 1
                        G[j, b] += g
                        H[j, b] += h

            # ─── Find best split ───
            best_gain = -1e9  # v5.0: Disable pre-pruning for post-pruning logic
            best_feature = np.int32(-1)
            best_bin = np.int32(0)

            for jj in range(n_selected_features):
                j = feature_indices[jj]
                nb = bin_counts[j]
                if nb < 2:
                    continue

                G_total = 0.0
                H_total = 0.0
                for b in range(nb):
                    G_total += G[j, b]
                    H_total += H[j, b]

                if is_categorical[j]:
                    mean_grad = np.zeros(nb, dtype=np.float64)
                    for b in range(nb):
                        if H[j, b] > 1e-9:
                            mean_grad[b] = G[j, b] / H[j, b]
                    sorted_bins = np.argsort(mean_grad)
                    G_L = 0.0
                    H_L = 0.0
                    for k in range(nb - 1):
                        b = sorted_bins[k]
                        G_L += G[j, b]
                        H_L += H[j, b]
                        G_R = G_total - G_L
                        H_R = H_total - H_L
                        if H_L < min_child_weight or H_R < min_child_weight:
                            continue
                        gain = (G_L * G_L / (H_L + reg_lambda) +
                                G_R * G_R / (H_R + reg_lambda) -
                                G_total * G_total / (H_total + reg_lambda))
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = np.int32(j)
                            bitset = np.uint64(0)
                            for i_bit in range(k + 1):
                                bitset |= (np.uint64(1) << np.uint64(sorted_bins[i_bit]))
                            best_bin = np.int64(bitset)
                else:
                    G_L = 0.0
                    H_L = 0.0
                    for b in range(nb - 1):
                        G_L += G[j, b]
                        H_L += H[j, b]
                        G_R = G_total - G_L
                        H_R = H_total - H_L
                        if H_L < min_child_weight or H_R < min_child_weight:
                            continue
                        gain = (G_L * G_L / (H_L + reg_lambda) +
                                G_R * G_R / (H_R + reg_lambda) -
                                G_total * G_total / (H_total + reg_lambda))
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = np.int32(j)
                            best_bin = np.int64(b)

            if best_feature < 0:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            # ─── In-place partition ───
            left_pos = start
            right_pos = end - 1
            is_cat = is_categorical[best_feature]
            mask = np.uint64(best_bin)
            while left_pos <= right_pos:
                i = indices[left_pos]
                val = X_bin[i, best_feature]
                if is_cat:
                    go_left = (mask >> np.uint64(val)) & np.uint64(1)
                else:
                    go_left = (val <= best_bin)
                if go_left:
                    left_pos += 1
                else:
                    indices[left_pos] = indices[right_pos]
                    indices[right_pos] = i
                    right_pos -= 1

            split_pos = left_pos
            n_left = split_pos - start
            n_right = end - split_pos

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            # Allocate children
            left_id = node_count; node_count += 1
            right_id = node_count; node_count += 1

            tree_features[cur_node] = best_feature
            tree_thresholds[cur_node] = best_bin
            tree_left[cur_node] = left_id
            tree_right[cur_node] = right_id
            tree_values[cur_node] = leaf_val  # v5.0: Save value for post-pruning fallback
            split_gains[cur_node] = best_gain

            # ─── HISTOGRAM SUBTRACTION TRICK ───
            # Determine smaller child, build its hist, subtract for larger
            if n_left <= n_right:
                smaller_start, smaller_end = start, split_pos
                larger_start, larger_end = split_pos, end
                smaller_id, larger_id = left_id, right_id
            else:
                smaller_start, smaller_end = split_pos, end
                larger_start, larger_end = start, split_pos
                smaller_id, larger_id = right_id, left_id

            # Build histogram for SMALLER child only
            # (We'll store it directly in the stack slot for the smaller child)
            smaller_slot = stack_ptr + 1  # will be pushed second (popped first)
            larger_slot = stack_ptr       # will be pushed first (popped second)

            # Build smaller child's histogram
            for jj in range(n_selected_features):
                j = feature_indices[jj]
                nb = bin_counts[j]
                for b in range(nb):
                    hist_G[smaller_slot, j, b] = 0.0
                    hist_H[smaller_slot, j, b] = 0.0

            for k in range(smaller_start, smaller_end):
                i = indices[k]
                g = grad[i]
                h = hess[i]
                for jj in range(n_selected_features):
                    j = feature_indices[jj]
                    b = X_bin[i, j]
                    if b >= bin_counts[j]:
                        b = bin_counts[j] - 1
                    hist_G[smaller_slot, j, b] += g
                    hist_H[smaller_slot, j, b] += h

            # Compute larger child's histogram by SUBTRACTION (FREE!)
            for jj in range(n_selected_features):
                j = feature_indices[jj]
                nb = bin_counts[j]
                for b in range(nb):
                    hist_G[larger_slot, j, b] = G[j, b] - hist_G[smaller_slot, j, b]
                    hist_H[larger_slot, j, b] = H[j, b] - hist_H[smaller_slot, j, b]

            # Push larger child first (popped second = processed second)
            stack_node[stack_ptr] = larger_id
            stack_start[stack_ptr] = larger_start
            stack_end[stack_ptr] = larger_end
            stack_depth[stack_ptr] = depth + 1
            stack_has_hist[stack_ptr] = 1
            stack_ptr += 1

            # Push smaller child second (popped first = processed first)
            stack_node[stack_ptr] = smaller_id
            stack_start[stack_ptr] = smaller_start
            stack_end[stack_ptr] = smaller_end
            stack_depth[stack_ptr] = depth + 1
            stack_has_hist[stack_ptr] = 1
            stack_ptr += 1

        return tree_features, tree_thresholds, tree_values, tree_left, tree_right, node_count, split_gains

    # ═══════════════════════════════════════════════════════════════════════
    # LEAF-WISE (BEST-FIRST) TREE BUILDER  —  v2.7
    # Like LightGBM: always splits the leaf with the highest gain.
    # Produces asymmetric, deeper trees that need fewer leaves for same
    # accuracy as a full level-wise tree — fewer histogram computations.
    # ═══════════════════════════════════════════════════════════════════════

    @njit(parallel=True, cache=True)
    def _build_tree_leafwise(X_bin, grad, hess, sample_indices, feature_indices,
                              max_leaves, max_depth, min_samples_leaf, min_child_weight,
                              bin_counts, max_bins, reg_lambda, gamma, n_all_features, is_categorical):
        """
        Leaf-wise (Best-First) tree growth.
        max_leaves: maximum number of leaf nodes (controls complexity).
        max_depth:  hard cap on depth (prevents overfitting on small datasets).
        """
        n_samples = len(sample_indices)
        n_selected_features = len(feature_indices)
        max_nodes = max_leaves * 2 + 2

        # Tree arrays
        tree_features  = np.full(max_nodes, -1, dtype=np.int32)
        tree_thresholds = np.zeros(max_nodes, dtype=np.int64)
        tree_values    = np.zeros(max_nodes, dtype=np.float64)
        tree_left      = np.full(max_nodes, -1, dtype=np.int32)
        tree_right     = np.full(max_nodes, -1, dtype=np.int32)
        split_gains_lw = np.zeros(max_nodes, dtype=np.float64)
        node_count     = np.int32(0)

        # Working copy of sample indices (partitioned in-place per leaf)
        indices = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            indices[i] = sample_indices[i]

        # ── Candidate leaf table ──
        # Each slot stores one potential leaf to split.
        MAX_C = max_leaves + 2

        # Histograms for all candidate leaves
        hist_G = np.zeros((MAX_C, n_all_features, max_bins), dtype=np.float64)
        hist_H = np.zeros((MAX_C, n_all_features, max_bins), dtype=np.float64)

        c_node   = np.zeros(MAX_C, dtype=np.int32)
        c_start  = np.zeros(MAX_C, dtype=np.int64)
        c_end    = np.zeros(MAX_C, dtype=np.int64)
        c_depth  = np.zeros(MAX_C, dtype=np.int32)
        c_gain   = np.full(MAX_C, -1.0, dtype=np.float64)
        c_feat   = np.full(MAX_C, -1, dtype=np.int32)
        c_bin    = np.zeros(MAX_C, dtype=np.int32)
        c_val    = np.zeros(MAX_C, dtype=np.float64)
        c_active = np.zeros(MAX_C, dtype=np.int8)
        n_splits_done = np.int32(0)

        # ── Root node initialisation ──
        root_id = node_count
        node_count += 1

        g_root = np.float64(0.0)
        h_root = np.float64(0.0)
        for k in range(n_samples):
            ii = indices[k]
            g_root += grad[ii]
            h_root += hess[ii]
        root_val = -g_root / (h_root + reg_lambda)

        if n_samples < min_samples_leaf or h_root < min_child_weight:
            tree_features[root_id] = -1
            tree_values[root_id]   = root_val
            return (tree_features, tree_thresholds, tree_values,
                    tree_left, tree_right, node_count, split_gains_lw)

        # Compute root histogram (v4.0: Parallelized via prange)
        for jj in prange(n_selected_features):
            j = feature_indices[jj]
            nb = bin_counts[j]
            for b in range(nb):
                hist_G[0, j, b] = 0.0
                hist_H[0, j, b] = 0.0
            for k in range(n_samples):
                ii = indices[k]
                g = grad[ii]
                h = hess[ii]
                bv = X_bin[ii, j]
                if bv >= nb:
                    bv = nb - 1
                hist_G[0, j, bv] += g
                hist_H[0, j, bv] += h

        # Find root best split
        r_best_gain = -1e9  # v5.0: Disable pre-pruning
        r_best_feat = np.int32(-1)
        r_best_bin  = np.int32(0)
        for jj in range(n_selected_features):
            j = feature_indices[jj]
            nb = bin_counts[j]
            if nb < 2:
                continue
            G_tot = np.float64(0.0); H_tot = np.float64(0.0)
            for b in range(nb):
                G_tot += hist_G[0, j, b]; H_tot += hist_H[0, j, b]
            GL = np.float64(0.0); HL = np.float64(0.0)
            for b in range(nb - 1):
                GL += hist_G[0, j, b]; HL += hist_H[0, j, b]
                GR = G_tot - GL; HR = H_tot - HL
                if HL < min_child_weight or HR < min_child_weight:
                    continue
                gain = (GL*GL/(HL+reg_lambda) + GR*GR/(HR+reg_lambda)
                        - G_tot*G_tot/(H_tot+reg_lambda))
                if gain > r_best_gain:
                    r_best_gain = gain
                    r_best_feat = np.int32(j)
                    r_best_bin  = np.int32(b)

        # Register root in candidate table at slot 0
        c_node[0]   = root_id
        c_start[0]  = np.int64(0)
        c_end[0]    = np.int64(n_samples)
        c_depth[0]  = np.int32(0)
        c_gain[0]   = r_best_gain if r_best_feat >= 0 else np.float64(-1.0)
        c_feat[0]   = r_best_feat
        c_bin[0]    = r_best_bin
        c_val[0]    = root_val
        c_active[0] = np.int8(1)

        if r_best_feat < 0:
            # Root cannot be split
            tree_features[root_id] = -1
            tree_values[root_id]   = root_val
            return (tree_features, tree_thresholds, tree_values,
                    tree_left, tree_right, node_count, split_gains_lw)

        # ── Main leaf-wise loop ──
        while n_splits_done < max_leaves - 1:

            # Linear scan: find candidate with max gain
            best_ci   = np.int32(-1)
            best_gain = gamma
            for ci in range(MAX_C):
                if c_active[ci] and c_feat[ci] >= 0 and c_gain[ci] > best_gain:
                    best_gain = c_gain[ci]
                    best_ci   = np.int32(ci)

            if best_ci < 0:
                break  # No more profitable splits

            ci           = best_ci
            cur_node     = c_node[ci]
            start        = c_start[ci]
            end          = c_end[ci]
            depth        = c_depth[ci]
            b_feat       = c_feat[ci]
            b_bin        = c_bin[ci]

            # Partition indices in-place
            lp = start
            rp = end - 1
            while lp <= rp:
                ii = indices[lp]
                if X_bin[ii, b_feat] <= b_bin:
                    lp += 1
                else:
                    indices[lp] = indices[rp]
                    indices[rp] = ii
                    rp -= 1
            split_pos = lp
            n_left    = split_pos - start
            n_right   = end - split_pos

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                # Mark as leaf (invalid split)
                tree_features[cur_node] = -1
                tree_values[cur_node]   = c_val[ci]
                c_active[ci]            = np.int8(0)
                continue

            # Assign child node IDs
            left_id  = node_count; node_count += 1
            right_id = node_count; node_count += 1

            tree_features[cur_node]   = b_feat
            tree_thresholds[cur_node] = b_bin
            tree_left[cur_node]       = left_id
            tree_right[cur_node]      = right_id
            tree_values[cur_node]     = c_val[ci]  # v5.0: Save value for post-pruning fallback
            split_gains_lw[cur_node]  = c_gain[ci]

            n_splits_done += 1
            new_depth = depth + 1

            # ── Compute stats for SMALLER child, subtract for LARGER ──
            # (histogram subtraction trick — halves histogram work)
            do_left_first = (n_left <= n_right)
            if do_left_first:
                small_start = start;     small_end = split_pos
                large_start = split_pos; large_end = end
                small_id = left_id;      large_id  = right_id
            else:
                small_start = split_pos; small_end = end
                large_start = start;     large_end = split_pos
                small_id = right_id;     large_id  = left_id

            # Find empty slots for the two new children
            slot_small = -1
            slot_large = -1
            for si in range(MAX_C):
                if si != ci and not c_active[si]:
                    if slot_small < 0:
                        slot_small = si
                    elif slot_large < 0:
                        slot_large = si
                        break

            # Build histogram for smaller child (v4.0: Parallelized via prange)
            for jj in prange(n_selected_features):
                j = feature_indices[jj]
                nb = bin_counts[j]
                for b in range(nb):
                    hist_G[slot_small, j, b] = 0.0
                    hist_H[slot_small, j, b] = 0.0
                for k in range(small_start, small_end):
                    ii = indices[k]
                    g = grad[ii]
                    h = hess[ii]
                    bv = X_bin[ii, j]
                    if bv >= nb:
                        bv = nb - 1
                    hist_G[slot_small, j, bv] += g
                    hist_H[slot_small, j, bv] += h

            # Build histogram for larger child by subtraction (FREE!)
            for jj in prange(n_selected_features):
                j = feature_indices[jj]
                nb = bin_counts[j]
                for b in range(nb):
                    hist_G[slot_large, j, b] = hist_G[ci, j, b] - hist_G[slot_small, j, b]
                    hist_H[slot_large, j, b] = hist_H[ci, j, b] - hist_H[slot_small, j, b]

            # Parent is now fully replaced
            c_active[ci] = np.int8(0)

            # Process BOTH children (smaller: use built hist; larger: parent - smaller)
            for pass_idx in range(2):
                if pass_idx == 0:
                    child_start = small_start; child_end = small_end
                    child_id    = small_id
                    child_n     = small_end - small_start
                    c_slot      = slot_small
                else:
                    child_start = large_start; child_end = large_end
                    child_id    = large_id
                    child_n     = large_end - large_start
                    c_slot      = slot_large

                # Compute leaf value for this child
                g_c = np.float64(0.0); h_c = np.float64(0.0)
                for k in range(child_start, child_end):
                    ii = indices[k]
                    g_c += grad[ii]
                    h_c += hess[ii]
                child_val = -g_c / (h_c + reg_lambda)

                child_best_gain = -1e9  # v5.0: Disable pre-pruning
                child_best_feat = np.int32(-1)
                child_best_bin  = np.int32(0)

                can_split = (child_n >= min_samples_leaf and
                             h_c >= min_child_weight and
                             new_depth < max_depth)

                if can_split:
                    for jj in range(n_selected_features):
                        j = feature_indices[jj]
                        nb = bin_counts[j]
                        if nb < 2:
                            continue
                        G_tot = np.float64(0.0); H_tot = np.float64(0.0)
                        for b in range(nb):
                            G_tot += hist_G[c_slot, j, b]
                            H_tot += hist_H[c_slot, j, b]
                        GL = np.float64(0.0); HL = np.float64(0.0)
                        for b in range(nb - 1):
                            GL += hist_G[c_slot, j, b]
                            HL += hist_H[c_slot, j, b]
                            GR = G_tot - GL; HR = H_tot - HL
                            if HL < min_child_weight or HR < min_child_weight:
                                continue
                            gain = (GL*GL/(HL+reg_lambda) + GR*GR/(HR+reg_lambda)
                                    - G_tot*G_tot/(H_tot+reg_lambda))
                            if gain > child_best_gain:
                                child_best_gain = gain
                                child_best_feat = np.int32(j)
                                child_best_bin  = np.int32(b)

                c_node[c_slot]   = child_id
                c_start[c_slot]  = np.int64(child_start)
                c_end[c_slot]    = np.int64(child_end)
                c_depth[c_slot]  = np.int32(new_depth)
                c_gain[c_slot]   = (child_best_gain
                                    if child_best_feat >= 0
                                    else np.float64(-1.0))
                c_feat[c_slot]   = child_best_feat
                c_bin[c_slot]    = child_best_bin
                c_val[c_slot]    = child_val
                c_active[c_slot] = np.int8(1)

        # Finalise all remaining candidate leaves as terminal nodes
        for ci in range(MAX_C):
            if c_active[ci]:
                nid = c_node[ci]
                tree_features[nid] = -1
                tree_values[nid]   = c_val[ci]

        return (tree_features, tree_thresholds, tree_values,
                tree_left, tree_right, node_count, split_gains_lw)

    # ─── Post-Pruning (v5.0) ───

    @njit(cache=True)
    def _post_prune_tree_jit(features, left, right, split_gains, gamma, node_count):
        # Bottom-up pruning without recursion
        # Parent ID is always less than child ID, so iterating backwards guarantees 
        # we process children before their parents.
        for node in range(node_count - 1, -1, -1):
            if features[node] != -1:  # is internal node
                l = left[node]
                r = right[node]
                # If both children are leaves
                if features[l] == -1 and features[r] == -1 and split_gains[node] < gamma:
                    # Prune!
                    features[node] = -1
                    left[node] = -1
                    right[node] = -1

    @njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
    def predict_batch_jit(X_bin, features, thresholds, values, left, right, is_categorical):
        n = X_bin.shape[0]
        preds = np.empty(n, dtype=np.float64)
        for i in prange(n):
            node = 0
            while features[node] >= 0:
                feat = features[node]
                val = X_bin[i, feat]
                if is_categorical[feat]:
                    go_left = (np.uint64(thresholds[node]) >> np.uint64(val)) & np.uint64(1)
                else:
                    go_left = (val <= thresholds[node])
                if go_left:
                    node = left[node]
                else:
                    node = right[node]
            preds[i] = values[node]
        return preds

    @njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
    def predict_all_trees_jit(X_bin, all_features, all_thresholds, all_values,
                               all_left, all_right, tree_offsets, n_trees, learning_rate, is_categorical):
        n = X_bin.shape[0]
        F = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            for t in range(n_trees):
                offset = tree_offsets[t]
                node = 0
                while all_features[offset + node] >= 0:
                    feat = all_features[offset + node]
                    val = X_bin[i, feat]
                    if is_categorical[feat]:
                        go_left = (np.uint64(all_thresholds[offset + node]) >> np.uint64(val)) & np.uint64(1)
                    else:
                        go_left = (val <= all_thresholds[offset + node])
                    if go_left:
                        node = all_left[offset + node]
                    else:
                        node = all_right[offset + node]
                F[i] += learning_rate * all_values[offset + node]
        return F

    @njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
    def predict_contributions_all_trees_jit(X_bin, all_features, all_thresholds, all_values,
                               all_left, all_right, tree_offsets, n_trees, learning_rate, is_categorical, n_features):
        n = X_bin.shape[0]
        # output shape: (n_samples, n_features + 1) where the last column is the expected value (bias)
        contributions = np.zeros((n, n_features + 1), dtype=np.float64)
        for i in prange(n):
            for t in range(n_trees):
                offset = tree_offsets[t]
                node = 0
                
                # The expected value of the tree is the value of the root node
                root_val = all_values[offset + 0]
                contributions[i, n_features] += learning_rate * root_val
                
                while all_features[offset + node] >= 0:
                    feat = all_features[offset + node]
                    val = X_bin[i, feat]
                    if is_categorical[feat]:
                        go_left = (np.uint64(all_thresholds[offset + node]) >> np.uint64(val)) & np.uint64(1)
                    else:
                        go_left = (val <= all_thresholds[offset + node])
                        
                    if go_left:
                        next_node = all_left[offset + node]
                    else:
                        next_node = all_right[offset + node]
                        
                    # Saabas heuristic TreeSHAP: contribution is the change in expected value
                    diff = learning_rate * (all_values[offset + next_node] - all_values[offset + node])
                    contributions[i, feat] += diff
                    
                    node = next_node
                    
        return contributions

else:
    # ─── Pure Python fallbacks ───
    def standard_binning_search(arr, target):
        low, high = 0, len(arr) - 2
        if target < arr[0]: return 0
        if target >= arr[-1]: return len(arr) - 2
        while low <= high:
            mid = (low + high) >> 1
            if arr[mid] <= target < arr[mid + 1]: return mid
            elif target < arr[mid]: high = mid - 1
            else: low = mid + 1
        return max(min(low, len(arr) - 2), 0)

    def advanced_binning_search(arr, target):
        low, high = 0, len(arr) - 1
        if target < arr[0]: return 0
        if target >= arr[high]: return high - 1
        while low + 1 <= high - 1:
            mid = ((low + 1) + (high - 1)) >> 1
            if arr[low] <= target < arr[low + 1]: return low
            if arr[high - 1] <= target < arr[high]: return high - 1
            if arr[mid] <= target < arr[mid + 1]: return mid
            if target < arr[mid]: high = mid - 1
            else: low = mid + 1
        return max(high - 1, 0)

    def hybrid_binning_numba(bins, values):
        n = len(values)
        n_bins = len(bins)
        indices = np.empty(n, dtype=np.int32)
        
        b_first = bins[0]
        last_idx = n_bins - 2
        b_end = bins[-1]
        scale = last_idx / (b_end - b_first) if b_end > b_first else 1.0
        
        for i in range(n):
            v = values[i]
            if v < b_first:
                indices[i] = 0
            elif v >= b_end:
                indices[i] = last_idx
            else:
                g = int((v - b_first) * scale)
                if g < 0:
                    g = 0
                elif g > last_idx:
                    g = last_idx
                
                if bins[g] <= v and (g == last_idx or v < bins[g + 1]):
                    indices[i] = g
                else:
                    lo = max(g - 4, 0)
                    hi = min(g + 4, last_idx)
                    
                    if v >= bins[lo] and (hi == last_idx or v < bins[hi + 1]):
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if bins[mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        indices[i] = lo
                    else:
                        lo = 0
                        hi = last_idx
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if bins[mid + 1] <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        indices[i] = lo
        return indices

    def standard_only_binning_numba(bins, values):
        n = len(values)
        indices = np.empty(n, dtype=np.int32)
        for i in range(n):
            indices[i] = standard_binning_search(bins, values[i])
        return indices

    def _build_tree_v25(X_bin, grad, hess, sample_indices, feature_indices,
                         max_depth, min_samples_leaf, min_child_weight,
                         bin_counts, max_bins, reg_lambda, gamma, n_all_features, is_categorical):
        """Pure Python fallback."""
        n_samples = len(sample_indices)
        n_sel = len(feature_indices)
        max_nodes = (1 << (max_depth + 1)) - 1

        tree_features = np.full(max_nodes, -1, dtype=np.int32)
        tree_thresholds = np.zeros(max_nodes, dtype=np.int64)
        tree_values = np.zeros(max_nodes, dtype=np.float64)
        tree_left = np.full(max_nodes, -1, dtype=np.int32)
        tree_right = np.full(max_nodes, -1, dtype=np.int32)
        split_gains = np.zeros(max_nodes, dtype=np.float64)

        indices = sample_indices.copy()
        stack = [(0, 0, n_samples, 0)]
        node_count = 1

        while stack:
            cur_node, start, end, depth = stack.pop()
            n_node = end - start
            g_sum = sum(grad[indices[k]] for k in range(start, end))
            h_sum = sum(hess[indices[k]] for k in range(start, end))
            leaf_val = -g_sum / (h_sum + reg_lambda)

            if depth >= max_depth or n_node < min_samples_leaf or h_sum < min_child_weight:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            G = np.zeros((n_all_features, max_bins))
            H = np.zeros((n_all_features, max_bins))
            for jj in range(n_sel):
                j = feature_indices[jj]
                nb = bin_counts[j]
                for k in range(start, end):
                    i = indices[k]
                    b = min(int(X_bin[i, j]), nb - 1)
                    G[j, b] += grad[i]
                    H[j, b] += hess[i]

            best_gain, best_feature, best_bin = gamma, -1, 0
            for jj in range(n_sel):
                j = feature_indices[jj]
                nb = bin_counts[j]
                if nb < 2: continue
                G_total = sum(G[j, :nb]); H_total = sum(H[j, :nb])
                G_L = H_L = 0.0
                for b in range(nb - 1):
                    G_L += G[j, b]; H_L += H[j, b]
                    G_R = G_total - G_L; H_R = H_total - H_L
                    if H_L < min_child_weight or H_R < min_child_weight: continue
                    gain = G_L**2/(H_L+reg_lambda) + G_R**2/(H_R+reg_lambda) - G_total**2/(H_total+reg_lambda)
                    if gain > best_gain:
                        best_gain, best_feature, best_bin = gain, j, b

            if best_feature < 0:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            left_pos, right_pos = start, end - 1
            while left_pos <= right_pos:
                i = indices[left_pos]
                if X_bin[i, best_feature] <= best_bin: left_pos += 1
                else:
                    indices[left_pos], indices[right_pos] = indices[right_pos], indices[left_pos]
                    right_pos -= 1

            n_left = left_pos - start; n_right = end - left_pos
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                tree_features[cur_node] = -1
                tree_values[cur_node] = leaf_val
                continue

            left_id = node_count; node_count += 1
            right_id = node_count; node_count += 1
            tree_features[cur_node] = best_feature
            tree_thresholds[cur_node] = best_bin
            tree_left[cur_node] = left_id
            tree_right[cur_node] = right_id
            split_gains[cur_node] = best_gain

            stack.append((left_id, start, left_pos, depth + 1))
            stack.append((right_id, left_pos, end, depth + 1))

        return tree_features, tree_thresholds, tree_values, tree_left, tree_right, node_count, split_gains


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA GPU HISTOGRAM BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_NUMBA:
    @cuda.jit
    def _build_histograms_cuda(X_bin, grad, hess, indices, feature_indices, 
                               node_start, node_end, hist_g, hist_h, is_categorical):
        """
        CUDA Kernel for computing histograms on the GPU using global memory atomics.
        Grid Strategy: 1D grid over samples. Each thread loops over selected features.
        """
        i = cuda.grid(1)
        if i >= (node_end - node_start):
            return
            
        # Get sample index
        idx = indices[node_start + i]
        
        # Thread loads its sample's grad and hess once
        g = grad[idx]
        h = hess[idx]
        
        # Loop over features
        for f_idx in range(len(feature_indices)):
            j = feature_indices[f_idx]
            if not is_categorical[j]:
                b = X_bin[idx, j]
                cuda.atomic.add(hist_g, (j, b), g)
                cuda.atomic.add(hist_h, (j, b), h)

    @cuda.jit
    def _build_histograms_cuda_cat(X_bin, grad, hess, indices, feature_indices, 
                                   node_start, node_end, hist_g, hist_h, is_categorical):
        """
        CUDA Kernel for computing histograms of categorical features.
        """
        i = cuda.grid(1)
        if i >= (node_end - node_start):
            return
            
        idx = indices[node_start + i]
        g = grad[idx]
        h = hess[idx]
        
        for f_idx in range(len(feature_indices)):
            j = feature_indices[f_idx]
            if is_categorical[j]:
                b = X_bin[idx, j]
                cuda.atomic.add(hist_g, (j, b), g)
                cuda.atomic.add(hist_h, (j, b), h)

    MAX_BINS_CUDA = 256
    MAX_CHUNK_FEATURES = 8

    @cuda.jit
    def _build_histograms_shared_cuda(X_bin, grad, hess, indices, feature_indices, 
                                      node_start, node_end, hist_g, hist_h, 
                                      chunk_start, chunk_end, is_categorical):
        """
        Advanced CUDA Kernel using Shared Memory (L1 Cache) Chunking.
        Reduces global memory atomic contention by aggregating locally in shared memory.
        """
        # Allocate shared memory (32 KB total)
        s_hist_g = cuda.shared.array(shape=(MAX_CHUNK_FEATURES, MAX_BINS_CUDA), dtype=numba.float64)
        s_hist_h = cuda.shared.array(shape=(MAX_CHUNK_FEATURES, MAX_BINS_CUDA), dtype=numba.float64)
        
        tid = cuda.threadIdx.x
        block_dim = cuda.blockDim.x
        
        # Step 1: Initialize shared memory to zero
        num_elements = MAX_CHUNK_FEATURES * MAX_BINS_CUDA
        for i in range(tid, num_elements, block_dim):
            f_idx_local = i // MAX_BINS_CUDA
            b_idx = i % MAX_BINS_CUDA
            s_hist_g[f_idx_local, b_idx] = 0.0
            s_hist_h[f_idx_local, b_idx] = 0.0
            
        cuda.syncthreads()
        
        # Step 2: Compute local histograms in shared memory
        i = cuda.grid(1)
        if i < (node_end - node_start):
            idx = indices[node_start + i]
            g = grad[idx]
            h = hess[idx]
            
            n_features_in_chunk = chunk_end - chunk_start
            for local_f in range(n_features_in_chunk):
                global_f_idx = chunk_start + local_f
                j = feature_indices[global_f_idx]
                if not is_categorical[j]:
                    b = X_bin[idx, j]
                    cuda.atomic.add(s_hist_g, (local_f, b), g)
                    cuda.atomic.add(s_hist_h, (local_f, b), h)
                    
        cuda.syncthreads()
        
        # Step 3: Flush shared memory back to global memory
        for i in range(tid, num_elements, block_dim):
            local_f = i // MAX_BINS_CUDA
            b_idx = i % MAX_BINS_CUDA
            if local_f < (chunk_end - chunk_start):
                global_f_idx = chunk_start + local_f
                j = feature_indices[global_f_idx]
                
                val_g = s_hist_g[local_f, b_idx]
                val_h = s_hist_h[local_f, b_idx]
                
                if val_g != 0.0 or val_h != 0.0:
                    cuda.atomic.add(hist_g, (j, b_idx), val_g)
                    cuda.atomic.add(hist_h, (j, b_idx), val_h)

    @cuda.jit
    def _build_histograms_shared_cuda_cat(X_bin, grad, hess, indices, feature_indices, 
                                          node_start, node_end, hist_g, hist_h, 
                                          chunk_start, chunk_end, is_categorical):
        """
        Advanced CUDA Kernel using Shared Memory Chunking for categorical features.
        """
        s_hist_g = cuda.shared.array(shape=(MAX_CHUNK_FEATURES, MAX_BINS_CUDA), dtype=numba.float64)
        s_hist_h = cuda.shared.array(shape=(MAX_CHUNK_FEATURES, MAX_BINS_CUDA), dtype=numba.float64)
        
        tid = cuda.threadIdx.x
        block_dim = cuda.blockDim.x
        
        num_elements = MAX_CHUNK_FEATURES * MAX_BINS_CUDA
        for i in range(tid, num_elements, block_dim):
            f_idx_local = i // MAX_BINS_CUDA
            b_idx = i % MAX_BINS_CUDA
            s_hist_g[f_idx_local, b_idx] = 0.0
            s_hist_h[f_idx_local, b_idx] = 0.0
            
        cuda.syncthreads()
        
        i = cuda.grid(1)
        if i < (node_end - node_start):
            idx = indices[node_start + i]
            g = grad[idx]
            h = hess[idx]
            
            n_features_in_chunk = chunk_end - chunk_start
            for local_f in range(n_features_in_chunk):
                global_f_idx = chunk_start + local_f
                j = feature_indices[global_f_idx]
                if is_categorical[j]:
                    b = X_bin[idx, j]
                    cuda.atomic.add(s_hist_g, (local_f, b), g)
                    cuda.atomic.add(s_hist_h, (local_f, b), h)
                    
        cuda.syncthreads()
        
        for i in range(tid, num_elements, block_dim):
            local_f = i // MAX_BINS_CUDA
            b_idx = i % MAX_BINS_CUDA
            if local_f < (chunk_end - chunk_start):
                global_f_idx = chunk_start + local_f
                j = feature_indices[global_f_idx]
                
                val_g = s_hist_g[local_f, b_idx]
                val_h = s_hist_h[local_f, b_idx]
                
                if val_g != 0.0 or val_h != 0.0:
                    cuda.atomic.add(hist_g, (j, b_idx), val_g)
                    cuda.atomic.add(hist_h, (j, b_idx), val_h)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
def _compute_lambdarank_gradients(y, F, group, sigma=1.0):
    n = len(y)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)
    n_groups = len(group)
    offsets = np.zeros(n_groups + 1, dtype=np.int32)
    for i in range(n_groups):
        offsets[i+1] = offsets[i] + group[i]
        
    for g in prange(n_groups):
        start = offsets[g]
        end = offsets[g+1]
        y_loc = y[start:end]
        F_loc = F[start:end]
        n_loc = end - start
        
        if n_loc < 2:
            continue
            
        idx = np.argsort(-F_loc)
        ideal_idx = np.argsort(-y_loc)
        idcg = 0.0
        for i in range(n_loc):
            rank = i + 1
            rel = y_loc[ideal_idx[i]]
            idcg += ((1 << int(rel)) - 1.0) / np.log2(rank + 1.0)
            
        if idcg == 0.0:
            continue
            
        inv_idcg = 1.0 / idcg
        discounts = np.zeros(n_loc, dtype=np.float64)
        for i in range(n_loc):
            discounts[i] = 1.0 / np.log2(i + 2.0)
            
        max_rank = min(n_loc, 15)
            
        for i in range(n_loc):
            idx_i = idx[i]
            y_i = y_loc[idx_i]
            for j in range(i + 1, n_loc):
                if i >= max_rank and j >= max_rank:
                    continue
                    
                idx_j = idx[j]
                y_j = y_loc[idx_j]
                
                if y_i == y_j:
                    continue
                    
                if y_i > y_j:
                    idx_high, idx_low = idx_i, idx_j
                    rank_high, rank_low = i, j
                else:
                    idx_high, idx_low = idx_j, idx_i
                    rank_high, rank_low = j, i
                    
                rel_high = y_loc[idx_high]
                rel_low = y_loc[idx_low]
                
                delta_dcg = (((1 << int(rel_high)) - 1.0) - ((1 << int(rel_low)) - 1.0)) * \
                            abs(discounts[rank_high] - discounts[rank_low])
                
                if delta_dcg == 0.0:
                    continue
                    
                delta_ndcg = delta_dcg * inv_idcg
                
                diff = F_loc[idx_high] - F_loc[idx_low]
                diff = max(min(diff, 50.0), -50.0)
                
                rho = 1.0 / (1.0 + np.exp(sigma * diff))
                
                lambda_ij = delta_ndcg * rho * sigma
                hess_ij = delta_ndcg * rho * (1.0 - rho) * sigma * sigma
                
                hess_ij = max(hess_ij, 1e-4)
                
                grad[start + idx_high] -= lambda_ij
                grad[start + idx_low] += lambda_ij
                hess[start + idx_high] += hess_ij
                hess[start + idx_low] += hess_ij
                
    return grad, hess


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA HOST ORCHESTRATION (HYBRID CPU-GPU TREE BUILDER)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_NUMBA:
    @njit(cache=True)
    def _partition_indices_njit(X_bin, indices, start, end, b_feat, b_bin, is_categorical):
        """Partition sample indices in-place around a split. Returns mid index."""
        lp = start
        rp = end - 1
        if not is_categorical[b_feat]:
            while lp <= rp:
                ii = indices[lp]
                if X_bin[ii, b_feat] <= b_bin:
                    lp += 1
                else:
                    indices[lp], indices[rp] = indices[rp], indices[lp]
                    rp -= 1
        else:
            while lp <= rp:
                ii = indices[lp]
                if X_bin[ii, b_feat] == b_bin:
                    lp += 1
                else:
                    indices[lp], indices[rp] = indices[rp], indices[lp]
                    rp -= 1
        return lp

    @njit(cache=True)
    def _find_best_split_njit(hist_G, hist_H, feature_indices, bin_counts, 
                              min_child_weight, reg_lambda, is_categorical):
        """Find the best split from a pre-computed histogram. Returns (feat, bin, gain)."""
        best_gain = -1e9
        best_feat = np.int32(-1)
        best_bin = np.int32(0)
        
        n_selected_features = len(feature_indices)
        for jj in range(n_selected_features):
            j = feature_indices[jj]
            nb = bin_counts[j]
            if nb < 2:
                continue
                
            G_tot = 0.0; H_tot = 0.0
            for b in range(nb):
                G_tot += hist_G[j, b]
                H_tot += hist_H[j, b]
                
            if is_categorical[j]:
                for b in range(nb):
                    GL = hist_G[j, b]
                    HL = hist_H[j, b]
                    GR = G_tot - GL
                    HR = H_tot - HL
                    if HL < min_child_weight or HR < min_child_weight:
                        continue
                    gain = (GL*GL/(HL+reg_lambda) + GR*GR/(HR+reg_lambda) 
                            - G_tot*G_tot/(H_tot+reg_lambda))
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = np.int32(j)
                        best_bin = np.int32(b)
            else:
                GL = 0.0; HL = 0.0
                for b in range(nb - 1):
                    GL += hist_G[j, b]
                    HL += hist_H[j, b]
                    GR = G_tot - GL
                    HR = H_tot - HL
                    if HL < min_child_weight or HR < min_child_weight:
                        continue
                    gain = (GL*GL/(HL+reg_lambda) + GR*GR/(HR+reg_lambda) 
                            - G_tot*G_tot/(H_tot+reg_lambda))
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = np.int32(j)
                        best_bin = np.int32(b)
                        
        return best_feat, best_bin, best_gain


def _build_tree_leafwise_cuda(X_bin, grad, hess, sample_indices, feature_indices,
                               max_leaves, max_depth, min_samples_leaf, min_child_weight,
                               bin_counts, max_bins, reg_lambda, gamma, n_all_features, 
                               is_categorical):
    """
    Leaf-wise tree builder using GPU-accelerated histogram construction.
    Uses Shared Memory Chunking CUDA kernels for histogram building,
    CPU Numba for split finding and index partitioning.
    """
    import math
    
    n_samples = len(sample_indices)
    n_selected_features = len(feature_indices)
    max_nodes = max_leaves * 2 + 2

    # Tree structure arrays (identical format to CPU builder)
    tree_features = np.full(max_nodes, -1, dtype=np.int32)
    tree_thresholds = np.zeros(max_nodes, dtype=np.int64)
    tree_values = np.zeros(max_nodes, dtype=np.float64)
    tree_left = np.full(max_nodes, -1, dtype=np.int32)
    tree_right = np.full(max_nodes, -1, dtype=np.int32)
    split_gains_lw = np.zeros(max_nodes, dtype=np.float64)
    
    indices = sample_indices.copy()
    
    MAX_C = max_leaves + 2
    c_node = np.zeros(MAX_C, dtype=np.int32)
    c_start = np.zeros(MAX_C, dtype=np.int64)
    c_end = np.zeros(MAX_C, dtype=np.int64)
    c_depth = np.zeros(MAX_C, dtype=np.int32)
    c_gain = np.full(MAX_C, -1.0, dtype=np.float64)
    c_feat = np.full(MAX_C, -1, dtype=np.int32)
    c_bin = np.zeros(MAX_C, dtype=np.int32)
    c_val = np.zeros(MAX_C, dtype=np.float64)
    c_active = np.zeros(MAX_C, dtype=np.int8)
    
    # ── Push static data to GPU VRAM (one-time transfer) ──
    d_X_bin = cuda.to_device(X_bin)
    d_grad = cuda.to_device(grad)
    d_hess = cuda.to_device(hess)
    d_feature_indices = cuda.to_device(feature_indices)
    d_is_categorical = cuda.to_device(is_categorical)
    
    CHUNK_SIZE = 8  # MAX_CHUNK_FEATURES
    threads_per_block = 256
    
    def compute_histogram_gpu(node_start, node_end, d_indices_local):
        """Launch shared memory CUDA kernels in feature chunks and return histograms."""
        hist_g = np.zeros((n_all_features, max_bins), dtype=np.float64)
        hist_h = np.zeros((n_all_features, max_bins), dtype=np.float64)
        d_hist_g = cuda.to_device(hist_g)
        d_hist_h = cuda.to_device(hist_h)
        
        n_node_samples = node_end - node_start
        blocks_per_grid = max(1, math.ceil(n_node_samples / threads_per_block))
        
        for chunk_start in range(0, n_selected_features, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_selected_features)
            _build_histograms_shared_cuda[blocks_per_grid, threads_per_block](
                d_X_bin, d_grad, d_hess, d_indices_local, d_feature_indices,
                node_start, node_end, d_hist_g, d_hist_h,
                chunk_start, chunk_end, d_is_categorical
            )
        cuda.synchronize()
        return d_hist_g.copy_to_host(), d_hist_h.copy_to_host()

    # ── Root node ──
    node_count = 1
    g_root = 0.0; h_root = 0.0
    for k in range(n_samples):
        ii = indices[k]
        g_root += grad[ii]
        h_root += hess[ii]
    root_val = -g_root / (h_root + reg_lambda)
    
    if n_samples < min_samples_leaf or h_root < min_child_weight:
        tree_features[0] = -1
        tree_values[0] = root_val
        return (tree_features, tree_thresholds, tree_values,
                tree_left, tree_right, node_count, split_gains_lw)

    # Build root histogram on GPU
    d_indices = cuda.to_device(indices)
    root_hist_g, root_hist_h = compute_histogram_gpu(0, n_samples, d_indices)
    
    # Find root best split on CPU
    best_feat, best_bin, best_gain = _find_best_split_njit(
        root_hist_g, root_hist_h, feature_indices, bin_counts,
        min_child_weight, reg_lambda, is_categorical
    )
    
    # Register root candidate
    c_node[0] = 0; c_start[0] = 0; c_end[0] = n_samples; c_depth[0] = 0
    c_gain[0] = float(best_gain); c_feat[0] = best_feat; c_bin[0] = best_bin
    c_val[0] = root_val; c_active[0] = 1

    if best_feat < 0:
        tree_features[0] = -1
        tree_values[0] = root_val
        return (tree_features, tree_thresholds, tree_values,
                tree_left, tree_right, node_count, split_gains_lw)

    # Store histograms for subtraction trick
    hist_G_store = np.zeros((MAX_C, n_all_features, max_bins), dtype=np.float64)
    hist_H_store = np.zeros((MAX_C, n_all_features, max_bins), dtype=np.float64)
    hist_G_store[0] = root_hist_g
    hist_H_store[0] = root_hist_h
    
    n_splits_done = 0
    
    # ── Main leaf-wise loop ──
    while n_splits_done < max_leaves - 1:
        # Find candidate with max gain
        best_ci = -1
        best_gain_val = float(gamma)
        for ci in range(MAX_C):
            if c_active[ci] and c_feat[ci] >= 0 and c_gain[ci] > best_gain_val:
                best_gain_val = c_gain[ci]
                best_ci = ci
                
        if best_ci < 0:
            break
            
        ci = best_ci
        cur_node = int(c_node[ci])
        start = int(c_start[ci])
        end = int(c_end[ci])
        depth = int(c_depth[ci])
        b_feat = int(c_feat[ci])
        b_bin = int(c_bin[ci])
        
        # Partition indices on CPU (fast Numba)
        mid = int(_partition_indices_njit(X_bin, indices, start, end, b_feat, b_bin, is_categorical))
        
        # Record split in tree
        tree_features[cur_node] = b_feat
        tree_thresholds[cur_node] = b_bin
        split_gains_lw[cur_node] = c_gain[ci]
        c_active[ci] = 0
        
        n_left = mid - start
        n_right = end - mid
        
        left_id = node_count; node_count += 1
        right_id = node_count; node_count += 1
        tree_left[cur_node] = left_id
        tree_right[cur_node] = right_id
        
        # Push updated partition to GPU
        d_indices = cuda.to_device(indices)
        
        # Build histogram for the SMALLER child on GPU (subtraction trick for the larger)
        if n_left <= n_right:
            small_hist_g, small_hist_h = compute_histogram_gpu(start, mid, d_indices)
            left_hist_g = small_hist_g
            left_hist_h = small_hist_h
            right_hist_g = hist_G_store[ci] - left_hist_g
            right_hist_h = hist_H_store[ci] - left_hist_h
        else:
            small_hist_g, small_hist_h = compute_histogram_gpu(mid, end, d_indices)
            right_hist_g = small_hist_g
            right_hist_h = small_hist_h
            left_hist_g = hist_G_store[ci] - right_hist_g
            left_hist_h = hist_H_store[ci] - right_hist_h
        
        # ── Left child ──
        gl_sum = 0.0; hl_sum = 0.0
        for jj in range(n_selected_features):
            j = feature_indices[jj]
            nb = bin_counts[j]
            for b in range(nb):
                gl_sum += left_hist_g[j, b]
                hl_sum += left_hist_h[j, b]
            break  # only need total from first pass
        # Recompute properly
        gl_sum = 0.0; hl_sum = 0.0
        for k in range(start, mid):
            ii = indices[k]
            gl_sum += grad[ii]
            hl_sum += hess[ii]
        l_val = -gl_sum / (hl_sum + reg_lambda)
        
        can_split_left = (depth + 1 < max_depth and n_left >= min_samples_leaf 
                          and hl_sum >= min_child_weight)
        if can_split_left:
            l_feat, l_bin, l_gain = _find_best_split_njit(
                left_hist_g, left_hist_h, feature_indices, bin_counts,
                min_child_weight, reg_lambda, is_categorical
            )
            if l_feat >= 0:
                c_node[ci] = left_id; c_start[ci] = start; c_end[ci] = mid
                c_depth[ci] = depth + 1; c_gain[ci] = float(l_gain)
                c_feat[ci] = l_feat; c_bin[ci] = l_bin; c_val[ci] = l_val
                c_active[ci] = 1
                hist_G_store[ci] = left_hist_g; hist_H_store[ci] = left_hist_h
            else:
                tree_features[left_id] = -1; tree_values[left_id] = l_val
        else:
            tree_features[left_id] = -1; tree_values[left_id] = l_val
            
        # ── Right child ──
        gr_sum = 0.0; hr_sum = 0.0
        for k in range(mid, end):
            ii = indices[k]
            gr_sum += grad[ii]
            hr_sum += hess[ii]
        r_val = -gr_sum / (hr_sum + reg_lambda)
        
        # Find empty slot for right child
        empty_slot = -1
        for i in range(MAX_C):
            if not c_active[i]:
                empty_slot = i
                break
                
        can_split_right = (empty_slot >= 0 and depth + 1 < max_depth 
                           and n_right >= min_samples_leaf and hr_sum >= min_child_weight)
        if can_split_right:
            r_feat, r_bin, r_gain = _find_best_split_njit(
                right_hist_g, right_hist_h, feature_indices, bin_counts,
                min_child_weight, reg_lambda, is_categorical
            )
            if r_feat >= 0:
                c_node[empty_slot] = right_id; c_start[empty_slot] = mid
                c_end[empty_slot] = end; c_depth[empty_slot] = depth + 1
                c_gain[empty_slot] = float(r_gain); c_feat[empty_slot] = r_feat
                c_bin[empty_slot] = r_bin; c_val[empty_slot] = r_val
                c_active[empty_slot] = 1
                hist_G_store[empty_slot] = right_hist_g
                hist_H_store[empty_slot] = right_hist_h
            else:
                tree_features[right_id] = -1; tree_values[right_id] = r_val
        else:
            tree_features[right_id] = -1; tree_values[right_id] = r_val
            
        n_splits_done += 1
        
    # Finalize remaining active candidates as leaf nodes
    for ci in range(MAX_C):
        if c_active[ci]:
            tree_features[c_node[ci]] = -1
            tree_values[c_node[ci]] = c_val[ci]
            
    return (tree_features, tree_thresholds, tree_values,
            tree_left, tree_right, node_count, split_gains_lw)


# ═══════════════════════════════════════════════════════════════════════
# GPU CUDA KERNELS (V2.0)
# ═══════════════════════════════════════════════════════════════════════
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    @cuda.jit
    def _build_histograms_cuda(X_bin, grad, hess, histograms, node_indices, n_samples, n_features):
        i = cuda.grid(1)
        if i < n_samples:
            node_id = node_indices[i]
            if node_id >= 0:
                g = grad[i]
                h = hess[i]
                for j in range(n_features):
                    bin_val = X_bin[i, j]
                    cuda.atomic.add(histograms, (node_id, j, bin_val, 0), g)
                    cuda.atomic.add(histograms, (node_id, j, bin_val, 1), h)

    @cuda.jit
    def _find_best_splits_cuda(histograms, split_results, node_id, n_features, max_bins, reg_lambda, gamma, min_child_weight):
        j = cuda.grid(1)
        if j < n_features:
            sum_g = 0.0
            sum_h = 0.0
            for b in range(max_bins):
                sum_g += histograms[node_id, j, b, 0]
                sum_h += histograms[node_id, j, b, 1]
            
            if sum_h < min_child_weight:
                return
                
            best_gain = -1.0
            best_bin = -1
            left_g = 0.0
            left_h = 0.0
            
            for b in range(max_bins - 1):
                left_g += histograms[node_id, j, b, 0]
                left_h += histograms[node_id, j, b, 1]
                right_g = sum_g - left_g
                right_h = sum_h - left_h
                
                if left_h < min_child_weight or right_h < min_child_weight:
                    continue
                    
                gain = (left_g**2 / (left_h + reg_lambda) + 
                        right_g**2 / (right_h + reg_lambda) - 
                        sum_g**2 / (sum_h + reg_lambda))
                        
                if gain > best_gain:
                    best_gain = gain
                    best_bin = b
            
            if best_gain > gamma:
                split_results[j, 0] = best_gain
                split_results[j, 1] = best_bin
                split_results[j, 2] = left_g
                split_results[j, 3] = left_h
                split_results[j, 4] = right_g
                split_results[j, 5] = right_h

    @cuda.jit
    def _partition_nodes_cuda(X_bin, node_indices, best_feature, best_bin, target_node_id, left_node_id, right_node_id, n_samples):
        i = cuda.grid(1)
        if i < n_samples:
            if node_indices[i] == target_node_id:
                if X_bin[i, best_feature] <= best_bin:
                    node_indices[i] = left_node_id
                else:
                    node_indices[i] = right_node_id

    @cuda.jit
    def _predict_all_trees_cuda(X_bin, features, thresholds, values, left, right, offsets, n_trees, learning_rate, out_preds, n_samples):
        i = cuda.grid(1)
        if i < n_samples:
            pred = 0.0
            for t in range(n_trees):
                node = offsets[t]
                while features[node] >= 0:
                    if X_bin[i, features[node]] <= thresholds[node]:
                        node = left[node]
                    else:
                        node = right[node]
                pred += learning_rate * values[node]
            out_preds[i] = pred


# ═══════════════════════════════════════════════════════════════════════
# CORE ESTIMATOR (CPU & GPU ROUTING)
# ═══════════════════════════════════════════════════════════════════════

class HybridHistGBMNumbaV1_2_0:
    """
    Blazing fast, Numba JIT-compiled Gradient Boosting Machine.
    Competition-grade Histogram GBM with Adaptive Hyperparameters.

    v4.0: Performance Parity Update (uint8 memory optimization, prange
    parallelization, and massive-dataset auto-tuning).

    Supports both binary and multi-class classification:
    - Binary (2 classes): Uses sigmoid, 1 tree per round.
    - Multi-class (K classes): Uses softmax, K trees per round.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or 'auto', default='auto'
    learning_rate : float or 'auto', default='auto'
    n_bins : int or 'auto', default='auto'
    reg_lambda : float, default=1.0
    min_samples_leaf : int, default=5
    min_child_weight : float, default=1.0
    gamma : float, default=0.0
    subsample : float, default=0.8
    colsample_bytree : float, default=0.8
    class_weight : str or None, default='balanced_sqrt'
        'balanced_sqrt' (recommended) uses sqrt-dampened weights for best
        Precision/Recall trade-off. 'balanced' uses full inverse-frequency
        weights (maximises Recall). None disables class weighting.
    monotone_constraints : list or None, default=None
        Per-feature monotonic constraint. Use 1 for increasing, -1 for
        decreasing, 0 for no constraint. Length must equal n_features.
        Example: [1, 0, -1] means feature 0 must increase target,
        feature 1 is unconstrained, feature 2 must decrease target.
    categorical_features : list or None, default=None
        List of column indices that should be treated as categorical.
        These features are split using gradient-sorted optimal grouping
        instead of standard threshold-based splits. No One-Hot Encoding
        needed — pass integer-encoded categories directly.
    early_stopping_rounds : int or None, default=None
    verbose : bool, default=True
    random_state : int or None, default=None
    device : str, default='cpu'
    """

    def __init__(self, n_estimators='auto', max_depth='auto', learning_rate='auto',
                 n_bins='auto', reg_lambda=1.0, min_samples_leaf=5,
                 min_child_weight='auto', gamma='auto',
                 subsample='auto', colsample_bytree='auto',
                 class_weight='balanced_sqrt',
                 grow_policy='leaf', max_leaves=31, binning_strategy='auto',
                 monotone_constraints=None, categorical_features=None,
                 early_stopping_rounds=None, verbose=True, random_state=None,
                 objective='auto', focal_gamma=2.0, loss='mse', device='cpu',
                 sampling_method='uniform', top_rate=0.2, other_rate=0.1,
                 scale_pos_weight=None):
        self.binning_strategy_init = binning_strategy
        self.n_estimators_init = n_estimators
        self.max_depth_init = max_depth
        self.learning_rate_init = learning_rate
        self.n_bins_init = n_bins
        self.reg_lambda = reg_lambda
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight_init = min_child_weight
        self.gamma_init = gamma
        self.subsample_init = subsample
        self.colsample_init = colsample_bytree
        self.device = device
        self.sampling_method = sampling_method  # 'uniform' (default) or 'goss'
        self.top_rate = top_rate                # GOSS: fraction of top-gradient samples to keep
        self.other_rate = other_rate            # GOSS: fraction of remaining samples to randomly keep
        self.scale_pos_weight = scale_pos_weight
        
        if self.device == 'cuda':
            if not HAS_NUMBA or not cuda.is_available():
                import warnings
                warnings.warn("CUDA is not available. Falling back to 'cpu' device.")
                self.device = 'cpu'
        self.class_weight = class_weight
        self.grow_policy = grow_policy          # 'leaf' or 'level'
        self.max_leaves = max_leaves            # v2.7: leaf-wise complexity control
        self.monotone_constraints = monotone_constraints  # v3.0
        self.categorical_features = categorical_features  # v3.0
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

        self.n_estimators = 100 if n_estimators == 'auto' else n_estimators
        self.max_depth = 5 if max_depth == 'auto' else max_depth
        self.learning_rate = 0.1 if learning_rate == 'auto' else learning_rate
        self.n_bins = 256 if n_bins == 'auto' else n_bins
        self.min_child_weight = 1.0 if min_child_weight == 'auto' else min_child_weight
        self.gamma = 0.0 if gamma == 'auto' else gamma
        self.subsample = 0.8 if subsample == 'auto' else subsample
        self.colsample_bytree = 0.8 if colsample_bytree == 'auto' else colsample_bytree
        self.objective = objective
        self.focal_gamma = focal_gamma
        self.loss = loss
        self.init_score_ = 0.0

        self.trees = []
        self.bin_edges = None
        self.feature_bin_counts = None
        self._max_bins = 0
        self.n_features_in_ = None

        # Multi-class attributes
        self.n_classes_ = 2
        self.classes_ = None
        self._is_multiclass = False
        # For multi-class: trees are stored as list of lists
        # self.mc_trees[class_k] = list of trees for class k
        self.mc_trees = None

        self.feature_importances_ = None
        self.feature_split_counts_ = None
        self.train_loss_history_ = []
        self.val_loss_history_ = []
        self.n_trees_fitted_ = 0
        self.best_iteration_ = 0

        self.times = {'binning': 0.0, 'training': 0.0, 'inference': 0.0}

        self._packed_features = None
        self._packed_thresholds = None
        self._packed_values = None
        self._packed_left = None
        self._packed_right = None
        self._tree_offsets = None
        # Multi-class packed trees: one set per class
        self._mc_packed = None

        # v2.9: Smart class weight auto-detection flags
        self._auto_no_weights = False
        self._auto_full_weights = False

        # v3.0: Internal state for categorical/missing features
        self._categorical_set = set()  # set of categorical column indices
        self._is_categorical = None    # boolean array per feature
        self._nan_bin_id = None        # per-feature NaN bin index (or -1)


    def _compute_adaptive_hyperparameters(self, X, y):
        n, p = X.shape
        K = self.n_classes_

        if self.binning_strategy_init == 'auto':
            # Adaptive binning switch: 'standard' is faster for dense/medium data.
            # 'hybrid' is safer/faster for extremely wide or massive datasets.
            if n > 20000 or p > 40:
                self.binning_strategy = 'hybrid'
            else:
                self.binning_strategy = 'standard'
        else:
            self.binning_strategy = self.binning_strategy_init

        if self.max_depth_init == 'auto':
            # Adaptive depth: deeper for more samples, shallower for few
            if K > 10:
                # Many-class needs deeper trees to discriminate
                self.max_depth = min(8, max(4, int(np.log2(max(n // 8, 8)))))
            else:
                self.max_depth = min(6, max(3, int(np.log2(max(n // 10, 8)))))

        if self.learning_rate_init == 'auto':
            data_std = np.std(X)
            # v2.12: Fine-tuned LR — lower for large datasets (more rounds compensate)
            if n > 20000:
                self.learning_rate = 0.08
            elif data_std > 2.0:     self.learning_rate = 0.10
            elif data_std > 1.5:   self.learning_rate = 0.12
            elif data_std > 1.0:   self.learning_rate = 0.15
            elif data_std > 0.5:   self.learning_rate = 0.18
            else:                  self.learning_rate = 0.20
            # Lower LR for many-class → compensated by more trees
            if K > 10:
                self.learning_rate *= 0.75
            elif K > 6:
                self.learning_rate *= 0.85

        if self.n_bins_init == 'auto':
            self.n_bins = min(256, p * 8 + 32)

        # ── v2.12: Adaptive n_estimators — more rounds for convergence ──
        if self.n_estimators_init == 'auto':
            if K <= 2:
                self.n_estimators = 150
            elif K <= 6:
                self.n_estimators = max(150, min(300, n // 10))
            elif K <= 15:
                # Many-class: need significantly more rounds
                self.n_estimators = max(200, min(400, n // 8))
            else:
                # Ultra many-class (Letter=26): max rounds for convergence
                self.n_estimators = max(250, min(500, n // 6))

        # ── v2.11: Gamma = 0 universally ──
        # max_depth and min_child_weight already prevent overfitting;
        # gamma was causing underfitting on Covertype/Letter.
        if self.gamma_init == 'auto':
            self.gamma = 0.0

        if self.min_child_weight_init == 'auto':
            if n > 20000:
                self.min_child_weight = 3.0
            elif n > 5000:
                self.min_child_weight = 2.0
            else:
                self.min_child_weight = 1.0

        # ── v2.12 & v4.0: Adaptive subsampling and L2 ──
        if self.subsample_init == 'auto':
            if n > 1000000:
                self.subsample = 0.5   # v4.0: SGB boosts generalization on massive data
            elif n > 20000:
                self.subsample = 0.75
            else:
                self.subsample = 0.8
                
        # v4.0: Scale L2 regularization for massive datasets to prevent overfitting
        if n > 1000000:
            self.reg_lambda = max(self.reg_lambda, 5.0)

        if self.colsample_init == 'auto':
            if p > 40:
                self.colsample_bytree = 0.7
            else:
                self.colsample_bytree = 0.8

        # ── v2.12: Adaptive max_leaves by dataset size ──
        if n > 10000:
            self.max_leaves = 63
        elif n < 2000:
            self.max_leaves = 15
        else:
            self.max_leaves = 31

        if self.verbose:
            print("  🔧 Adaptive Hyperparameters:")
            print(f"     n_estimators:  {self.n_estimators}")
            print(f"     max_depth:     {self.max_depth} (from {n} samples)")
            print(f"     learning_rate: {self.learning_rate:.4f} (data_std={np.std(X):.2f})")
            print(f"     n_bins:        {self.n_bins} (from {p} features)")
            print(f"     gamma:         {self.gamma:.2f}")
            print(f"     min_child_w:   {self.min_child_weight:.1f}")
            print(f"     subsample:     {self.subsample}")
            print(f"     colsample:     {self.colsample_bytree}")

    def _encode_categorical(self, X, is_fit=False):
        """Pre-processes categorical features by mapping them to integers 0..63."""
        X_enc = X.copy()
        if is_fit:
            self._cat_maps = {}
            for j in self.categorical_features:
                col = X[:, j]
                try:
                    nan_mask = pd.isna(col)
                except:
                    nan_mask = col != col
                vals, counts = np.unique(col[~nan_mask], return_counts=True)
                sort_idx = np.argsort(-counts)
                vals = vals[sort_idx]
                
                # Keep top 63 categories (bin 63 is reserved for "Other"/Missing)
                top_vals = vals[:63]
                mapping = {val: i for i, val in enumerate(top_vals)}
                self._cat_maps[j] = mapping
                
                mapped_col = np.full(len(col), 63, dtype=np.float64)
                for val, idx in mapping.items():
                    mapped_col[col == val] = idx
                mapped_col[nan_mask] = 63
                X_enc[:, j] = mapped_col
        else:
            for j in self.categorical_features:
                col = X[:, j]
                mapping = self._cat_maps[j]
                mapped_col = np.full(len(col), 63, dtype=np.float64)
                for val, idx in mapping.items():
                    mapped_col[col == val] = idx
                try:
                    nan_mask = pd.isna(col)
                except:
                    nan_mask = col != col
                mapped_col[nan_mask] = 63
                X_enc[:, j] = mapped_col
        return X_enc

    def _build_bin_edges(self, X, sample_weight=None):
        """Build bin edges for each feature, with native NaN support (v3.0).
        
        If a feature has NaN values, bin index 0 is reserved for NaN.
        Normal values are binned starting from index 1.
        self._nan_bin_id[j] == 0 if feature j has NaNs, else -1.
        """
        edges = []
        n_features = X.shape[1]
        bin_counts = np.zeros(n_features, dtype=np.int32)
        self._nan_bin_id = np.full(n_features, -1, dtype=np.int32)

        for j in range(n_features):
            if self.categorical_features is not None and j in self.categorical_features:
                # Exact integer bins 0..63 for categorical features
                edges.append(np.arange(-0.5, 64.5, 1.0))
                bin_counts[j] = 64
                continue

            col = X[:, j]
            has_nan = np.any(np.isnan(col))

            if has_nan:
                self._nan_bin_id[j] = 0  # bin 0 = NaN bin
                col_clean = col[~np.isnan(col)]
                if len(col_clean) == 0:
                    # All NaN — single bin for NaN
                    e = np.array([0.0, 1.0])
                    edges.append(e)
                    bin_counts[j] = 1  # only the NaN bin
                    continue
            else:
                col_clean = col

            n_unique = len(np.unique(col_clean))
            max_numeric_bins = self.n_bins - (1 if has_nan else 0)

            if n_unique <= max_numeric_bins:
                e = np.sort(np.unique(col_clean))
            else:
                q = np.linspace(0, 1, min(max_numeric_bins + 1, n_unique))
                if sample_weight is not None:
                    if has_nan:
                        sw_clean = sample_weight[~np.isnan(col)]
                    else:
                        sw_clean = sample_weight
                    sorter = np.argsort(col_clean)
                    c_sorted = col_clean[sorter]
                    sw_sorted = sw_clean[sorter]
                    cw = np.cumsum(sw_sorted) - 0.5 * sw_sorted
                    cw /= np.sum(sw_sorted)
                    e = np.unique(np.interp(q, cw, c_sorted))
                else:
                    e = np.unique(np.percentile(col_clean, q * 100))
            e = np.append(e, e[-1] + 1e-8)
            edges.append(e)

            if has_nan:
                bin_counts[j] = len(e) - 1 + 1  # +1 for the NaN bin at index 0
            else:
                bin_counts[j] = len(e) - 1

        max_b = int(np.max(bin_counts)) + 1
        edges_2d = np.zeros((n_features, max_b), dtype=np.float64)
        for j, e in enumerate(edges):
            edges_2d[j, :len(e)] = e
            edges_2d[j, len(e):] = e[-1]

        return edges_2d, bin_counts


    def _bin_features(self, X, edges=None):
        """Bin features with native NaN support (v3.0).
        
        For features with NaN: bin 0 = NaN, bins 1+ = normal values.
        For features without NaN: bins 0+ = normal values (unchanged).
        """
        if edges is None:
            edges = self.bin_edges
        t0 = time.perf_counter()

        # Check if any features have NaN
        has_any_nan = self._nan_bin_id is not None and np.any(self._nan_bin_id >= 0)

        if has_any_nan:
            # Replace NaN with a placeholder value for binning, then fix up
            nan_mask = np.isnan(X)
            X_safe = X.copy()
            X_safe[nan_mask] = 0.0  # Temporary — will be overwritten

            if self._max_bins <= 255:
                X_binned = np.empty((X.shape[0], X.shape[1]), dtype=np.uint8)
            else:
                X_binned = np.empty((X.shape[0], X.shape[1]), dtype=np.uint16)

            if self.binning_strategy == 'standard':
                X_binned = bin_all_features_standard_numba(X_safe, edges, self.feature_bin_counts, X_binned)
            else:
                X_binned = bin_all_features_hybrid_numba(X_safe, edges, self.feature_bin_counts, X_binned)

            # For NaN-aware features: offset normal bins by +1, set NaN positions to 0
            for j in range(X.shape[1]):
                if self._nan_bin_id[j] >= 0:
                    X_binned[:, j] += 1  # shift normal bins to 1+
                    X_binned[nan_mask[:, j], j] = 0  # NaN → bin 0
        else:
            if self._max_bins <= 255:
                X_binned = np.empty((X.shape[0], X.shape[1]), dtype=np.uint8)
            else:
                X_binned = np.empty((X.shape[0], X.shape[1]), dtype=np.uint16)

            if self.binning_strategy == 'standard':
                X_binned = bin_all_features_standard_numba(X, edges, self.feature_bin_counts, X_binned)
            else:
                X_binned = bin_all_features_hybrid_numba(X, edges, self.feature_bin_counts, X_binned)

        return X_binned, (time.perf_counter() - t0) * 1000


    def _pack_tree_list(self, tree_list):
        """Pack a list of trees into contiguous arrays for JIT prediction."""
        total_nodes = sum(t[5] for t in tree_list)
        features = np.empty(total_nodes, dtype=np.int32)
        thresholds = np.empty(total_nodes, dtype=np.int64)
        values = np.empty(total_nodes, dtype=np.float64)
        left = np.empty(total_nodes, dtype=np.int32)
        right = np.empty(total_nodes, dtype=np.int32)
        offsets = np.empty(len(tree_list) + 1, dtype=np.int32)

        offset = 0
        for t_idx, (tf, tt, tv, tl, tr, nc, _) in enumerate(tree_list):
            offsets[t_idx] = offset
            features[offset:offset + nc] = tf[:nc]
            thresholds[offset:offset + nc] = tt[:nc]
            values[offset:offset + nc] = tv[:nc]
            left[offset:offset + nc] = tl[:nc]
            right[offset:offset + nc] = tr[:nc]
            offset += nc
        offsets[len(tree_list)] = offset
        return {
            'features': features, 'thresholds': thresholds,
            'values': values, 'left': left, 'right': right,
            'offsets': offsets
        }

    def _pack_trees(self):
        """Pack trees for fast JIT prediction."""
        if self._is_multiclass:
            # Pack each class's trees separately
            self._mc_packed = []
            for k in range(self.n_classes_):
                if len(self.mc_trees[k]) > 0:
                    self._mc_packed.append(self._pack_tree_list(self.mc_trees[k]))
                else:
                    self._mc_packed.append(None)
        else:
            # Binary: pack into single set of arrays
            packed = self._pack_tree_list(self.trees)
            self._packed_features = packed['features']
            self._packed_thresholds = packed['thresholds']
            self._packed_values = packed['values']
            self._packed_left = packed['left']
            self._packed_right = packed['right']
            self._tree_offsets = packed['offsets']

    def _warmup_jit(self, X_bin):
        if not HAS_NUMBA:
            return
        n_feat = X_bin.shape[1]
        dummy_grad = np.zeros(10, dtype=np.float64)
        dummy_hess = np.ones(10, dtype=np.float64)
        dummy_samp = np.arange(min(10, X_bin.shape[0]), dtype=np.int64)
        dummy_feat = np.arange(n_feat, dtype=np.int32)

        # Warm up level-wise builder
        dummy_is_cat = np.zeros(n_feat, dtype=np.bool_)
        _build_tree_v25(
            X_bin[:10], dummy_grad, dummy_hess, dummy_samp[:2], dummy_feat,
            np.int32(2), np.int32(2), np.float64(1.0),
            self.feature_bin_counts, np.int32(self._max_bins),
            np.float64(1.0), np.float64(0.0), np.int32(n_feat), dummy_is_cat
        )

        # Warm up leaf-wise builder
        _build_tree_leafwise(
            X_bin[:10], dummy_grad, dummy_hess, dummy_samp[:2], dummy_feat,
            np.int32(4), np.int32(6), np.int32(2), np.float64(1.0),
            self.feature_bin_counts, np.int32(self._max_bins),
            np.float64(1.0), np.float64(0.0), np.int32(n_feat), dummy_is_cat
        )

        dummy_f = np.full(1, -1, dtype=np.int32)
        dummy_t = np.zeros(1, dtype=np.int32)
        dummy_v = np.zeros(1, dtype=np.float64)
        dummy_l = np.full(1, -1, dtype=np.int32)
        dummy_r = np.full(1, -1, dtype=np.int32)
        predict_batch_jit(X_bin[:2], dummy_f, dummy_t, dummy_v, dummy_l, dummy_r, dummy_is_cat)

        offsets = np.array([0, 1], dtype=np.int32)
        predict_all_trees_jit(X_bin[:2], dummy_f, dummy_t, dummy_v, dummy_l, dummy_r,
                               offsets, np.int32(1), np.float64(0.1), dummy_is_cat)
                               
        dummy_y = np.array([1.0, 0.0], dtype=np.float64)
        dummy_F = np.array([0.5, 0.5], dtype=np.float64)
        dummy_group = np.array([2], dtype=np.int32)
        _compute_lambdarank_gradients(dummy_y, dummy_F, dummy_group, 1.0)

    @staticmethod
    def _compute_ndcg(y, F, group):
        """Compute average NDCG over groups."""
        if group is None: return 0.0
        n_groups = len(group)
        offsets = np.zeros(n_groups + 1, dtype=np.int32)
        for i in range(n_groups):
            offsets[i+1] = offsets[i] + group[i]
            
        total_ndcg = 0.0
        valid_groups = 0
        for g in range(n_groups):
            start = offsets[g]
            end = offsets[g+1]
            y_loc = y[start:end]
            F_loc = F[start:end]
            
            if len(y_loc) < 2:
                continue
                
            idx = np.argsort(-F_loc)
            ideal_idx = np.argsort(-y_loc)
            
            dcg = 0.0
            idcg = 0.0
            for i in range(len(y_loc)):
                rank = i + 1
                rel = y_loc[idx[i]]
                ideal_rel = y_loc[ideal_idx[i]]
                
                dcg += ((1 << int(rel)) - 1.0) / np.log2(rank + 1.0)
                idcg += ((1 << int(ideal_rel)) - 1.0) / np.log2(rank + 1.0)
                
            if idcg > 0.0:
                total_ndcg += dcg / idcg
                valid_groups += 1
                
        if valid_groups == 0: return 0.0
        return total_ndcg / valid_groups

    @staticmethod
    def _compute_logloss(y, F):
        """Binary log-loss."""
        p = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    @staticmethod
    def _softmax(F):
        """Compute softmax probabilities. F shape: (n_samples, n_classes)."""
        F_shifted = F - F.max(axis=1, keepdims=True)  # numerical stability
        exp_F = np.exp(F_shifted)
        return exp_F / exp_F.sum(axis=1, keepdims=True)

    @staticmethod
    def _compute_multiclass_logloss(y_onehot, F):
        """Multi-class log-loss using softmax."""
        F_shifted = F - F.max(axis=1, keepdims=True)
        exp_F = np.exp(F_shifted)
        proba = exp_F / exp_F.sum(axis=1, keepdims=True)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_onehot * np.log(proba), axis=1))

    def _compute_feature_importances(self):
        gain_imp = np.zeros(self.n_features_in_, dtype=np.float64)
        split_cnt = np.zeros(self.n_features_in_, dtype=np.int64)

        def _accumulate(tree_list):
            for tf, tt, tv, tl, tr, nc, sg in tree_list:
                for node_idx in range(nc):
                    feat = tf[node_idx]
                    if feat >= 0:
                        gain_imp[feat] += sg[node_idx]
                        split_cnt[feat] += 1

        if self._is_multiclass:
            for class_trees in self.mc_trees:
                _accumulate(class_trees)
        else:
            _accumulate(self.trees)

        total = gain_imp.sum()
        self.feature_importances_ = gain_imp / total if total > 0 else np.zeros(self.n_features_in_)
        self.feature_split_counts_ = split_cnt

    def _predict_tree(self, X_bin, tf, tt, tv, tl, tr, nc):
        """Predict using a single tree (JIT or fallback)."""
        if HAS_NUMBA:
            return predict_batch_jit(X_bin, tf[:nc], tt[:nc], tv[:nc], tl[:nc], tr[:nc], self._get_is_categorical(X_bin.shape[1]))
        else:
            pred = np.empty(len(X_bin), dtype=np.float64)
            for i in range(len(X_bin)):
                node = 0
                while tf[node] >= 0:
                    if X_bin[i, tf[node]] <= tt[node]: node = tl[node]
                    else: node = tr[node]
                pred[i] = tv[node]
            return pred

    def _build_one_tree(self, X_bin, grad, hess, sample_idx, feat_idx, n_features):
        """Build a single tree using the configured grow_policy.
        
        v3.0: Applies monotonic constraint enforcement after tree building.
        """
        is_categorical = np.zeros(n_features, dtype=np.bool_)
        if self.categorical_features is not None:
            for j in self.categorical_features:
                is_categorical[j] = True
                
        if HAS_NUMBA and self.grow_policy == 'leaf':
            if self.device == 'cuda':
                tree = _build_tree_leafwise_cuda(
                    X_bin, grad, hess, sample_idx, feat_idx,
                    np.int32(self.max_leaves),
                    np.int32(self.max_depth),
                    np.int32(self.min_samples_leaf),
                    np.float64(self.min_child_weight),
                    self.feature_bin_counts,
                    np.int32(self._max_bins),
                    np.float64(self.reg_lambda),
                    np.float64(self.gamma),
                    np.int32(n_features),
                    is_categorical
                )
            else:
                tree = _build_tree_leafwise(
                    X_bin, grad, hess, sample_idx, feat_idx,
                    np.int32(self.max_leaves),
                    np.int32(self.max_depth),
                    np.int32(self.min_samples_leaf),
                    np.float64(self.min_child_weight),
                self.feature_bin_counts,
                np.int32(self._max_bins),
                np.float64(self.reg_lambda),
                np.float64(self.gamma),
                np.int32(n_features),
                is_categorical
            )
        else:
            tree = _build_tree_v25(
                X_bin, grad, hess, sample_idx, feat_idx,
                np.int32(self.max_depth),
                np.int32(self.min_samples_leaf),
                np.float64(self.min_child_weight),
                self.feature_bin_counts,
                np.int32(self._max_bins),
                np.float64(self.reg_lambda),
                np.float64(self.gamma),
                np.int32(n_features),
                is_categorical
            )

        # v5.0: True post-pruning
        tf, tt, tv, tl, tr, nc, sg = tree
        _post_prune_tree_jit(tf, tl, tr, sg, np.float64(self.gamma), nc)

        # v3.0: Enforce monotonic constraints post-hoc
        if self._mono_arr is not None:
            tf, tt, tv, tl, tr, nc, sg = tree
            self._enforce_monotonicity(tf, tt, tv, tl, tr, nc)

        return tree

    def _enforce_monotonicity(self, tf, tt, tv, tl, tr, nc):
        """Post-hoc monotonic constraint enforcement (v3.0).
        
        For each internal node split on a constrained feature:
        - If constraint is +1 (increasing): left child value must be <= right child value
        - If constraint is -1 (decreasing): left child value must be >= right child value
        Clip leaf values to satisfy the constraint.
        """
        if self._mono_arr is None:
            return

        for node_idx in range(nc):
            feat = tf[node_idx]
            if feat < 0:
                continue  # leaf node

            constraint = self._mono_arr[feat]
            if constraint == 0:
                continue  # unconstrained

            left_id = tl[node_idx]
            right_id = tr[node_idx]
            if left_id < 0 or right_id < 0:
                continue

            left_val = tv[left_id]
            right_val = tv[right_id]

            if constraint == 1:
                # Increasing: left (low values) should predict <= right (high values)
                if left_val > right_val:
                    mid = (left_val + right_val) / 2.0
                    tv[left_id] = mid
                    tv[right_id] = mid
            elif constraint == -1 and left_val < right_val:
                # Decreasing: left (low values) should predict >= right (high values)
                mid = (left_val + right_val) / 2.0
                tv[left_id] = mid
                tv[right_id] = mid


    def _get_subsample_indices(self, rng, all_sample_indices, all_feature_indices,
                               n_sub_samples, n_sub_features):
        """Generate stochastic row and column subsample indices."""
        if self.subsample < 1.0:
            rng.shuffle(all_sample_indices)
            sample_idx = all_sample_indices[:n_sub_samples].copy()
        else:
            sample_idx = all_sample_indices.copy()

        if self.colsample_bytree < 1.0:
            rng.shuffle(all_feature_indices)
            feat_idx = np.sort(all_feature_indices[:n_sub_features]).copy()
        else:
            feat_idx = all_feature_indices.copy()

        return sample_idx, feat_idx

    def _get_goss_indices(self, rng, grad, all_feature_indices, n_sub_features):
        """Gradient-based One-Side Sampling (GOSS) — LightGBM's speed trick.
        
        Instead of random row subsampling, GOSS keeps:
          1. The top `top_rate` fraction of samples with largest |gradient|
          2. A random `other_rate` fraction of the remaining samples
        
        The randomly sampled rows have their gradients multiplied by a
        compensation factor to maintain an unbiased gradient estimate.
        
        Returns: (sample_idx, feat_idx, grad_weight)
          grad_weight is a per-sample multiplier array (1.0 for top, 
          compensation factor for others). Applied to grad/hess before
          tree building.
        """
        n_samples = len(grad)
        abs_grad = np.abs(grad)
        
        # Sort by |gradient| descending
        sorted_indices = np.argsort(-abs_grad)
        
        n_top = max(1, int(n_samples * self.top_rate))
        n_other = max(1, int(n_samples * self.other_rate))
        
        # Top gradient samples (always kept)
        top_indices = sorted_indices[:n_top]
        
        # Randomly sample from the rest
        rest_indices = sorted_indices[n_top:]
        rng.shuffle(rest_indices)
        other_indices = rest_indices[:n_other]
        
        # Combine
        selected = np.concatenate([top_indices, other_indices])
        
        # Compensation weight: the "other" samples represent
        # (1 - top_rate) / other_rate times their actual count
        compensation = (1.0 - self.top_rate) / max(self.other_rate, 1e-6)
        grad_weight = np.ones(n_samples, dtype=np.float64)
        grad_weight[other_indices] = compensation
        
        # Column subsampling (same as uniform)
        if self.colsample_bytree < 1.0:
            rng.shuffle(all_feature_indices)
            feat_idx = np.sort(all_feature_indices[:n_sub_features]).copy()
        else:
            feat_idx = all_feature_indices.copy()
        
        sample_idx = selected.astype(np.int64)
        return sample_idx, feat_idx, grad_weight

    def _compute_class_weights(self, y, n_samples):
        """Compute class weights with optional sqrt dampening."""
        if self.scale_pos_weight is not None and self.objective == 'binary':
            sample_weight = np.ones(n_samples, dtype=np.float64)
            sample_weight[y == 1] = self.scale_pos_weight
            if self.verbose:
                print(f"  ⚖️  Manual scale_pos_weight applied: positive_weight={self.scale_pos_weight}")
            return sample_weight
            
        if self.objective in ('regression', 'lambdarank') or self.class_weight not in ('balanced', 'balanced_sqrt'):
            return np.ones(n_samples, dtype=np.float64)
            
        sample_weight = np.ones(n_samples, dtype=np.float64)
        K = self.n_classes_
        use_sqrt = (self.class_weight == 'balanced_sqrt')
        if self.verbose:
            weights_str = []
        for k in range(K):
            n_k = np.sum(y == k)
            w_k = n_samples / (K * max(n_k, 1))
            if use_sqrt:
                # v2.11: Power-0.35 dampening (softer than sqrt=0.5)
                # Keeps majority-class precision while still boosting
                # minority recall. Dramatically helps Covertype/Letter.
                w_k = w_k ** 0.35
            
            sample_weight[y == k] = w_k
            
            if self.verbose:
                weights_str.append(f"class_{k}={w_k:.3f}")
        if self.verbose:
            mode = "sqrt-dampened" if use_sqrt else "full"
            print(f"  ⚖️  Class weights ({mode}): {', '.join(weights_str)}")
        return sample_weight

    # ─────────────────────────────────────────────────────────────────────
    # REGRESSION TRAINING
    # ─────────────────────────────────────────────────────────────────────
    def _fit_regression(self, X_bin, y, X_val_bin, y_val, rng):
        """Regression training loop (MSE/MAE, 1 tree/round)."""
        n_samples = len(y)
        n_features = X_bin.shape[1]
        
        self.init_score_ = float(np.mean(y))
        F_train = np.full(n_samples, self.init_score_, dtype=np.float64)
        F_val = np.full(len(y_val), self.init_score_, dtype=np.float64) if y_val is not None else None

        n_sub_samples = max(1, int(n_samples * self.subsample))
        n_sub_features = max(1, int(n_features * self.colsample_bytree))

        best_val_loss = np.inf
        rounds_no_improve = 0
        all_sample_indices = np.arange(n_samples, dtype=np.int64)
        all_feature_indices = np.arange(n_features, dtype=np.int32)

        for m in range(self.n_estimators):
            if self.loss == 'mse':
                grad = (F_train - y).astype(np.float64)
                hess = np.ones(n_samples, dtype=np.float64)
            elif self.loss == 'mae':
                grad = np.sign(F_train - y).astype(np.float64)
                hess = np.ones(n_samples, dtype=np.float64)

            if self.sampling_method == 'goss':
                sample_idx, feat_idx, grad_weight = self._get_goss_indices(
                    rng, grad, all_feature_indices, n_sub_features
                )
                grad *= grad_weight
                hess *= grad_weight
            else:
                sample_idx, feat_idx = self._get_subsample_indices(
                    rng, all_sample_indices, all_feature_indices,
                    n_sub_samples, n_sub_features
                )

            tf, tt, tv, tl, tr, nc, sg = self._build_one_tree(
                X_bin, grad, hess, sample_idx, feat_idx, n_features
            )
            self.trees.append((tf, tt, tv, tl, tr, nc, sg))

            tree_pred = self._predict_tree(X_bin, tf, tt, tv, tl, tr, nc)
            F_train += self.learning_rate * tree_pred

            train_loss = np.mean((y - F_train) ** 2) if self.loss == 'mse' else np.mean(np.abs(y - F_train))
            self.train_loss_history_.append(train_loss)

            val_loss = None
            if X_val_bin is not None:
                val_pred = self._predict_tree(X_val_bin, tf, tt, tv, tl, tr, nc)
                F_val += self.learning_rate * val_pred
                val_loss = np.mean((y_val - F_val) ** 2) if self.loss == 'mse' else np.mean(np.abs(y_val - F_val))
                self.val_loss_history_.append(val_loss)

            if self.verbose and ((m + 1) % 20 == 0 or m == 0):
                msg = f"  ✓ Tree {m+1:>3}/{self.n_estimators}  |  train-loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f"  |  val-loss: {val_loss:.6f}"
                print(msg)

            if self.early_stopping_rounds is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_iteration_ = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"\n  ⏹ Early stopping at round {m+1} (best: {self.best_iteration_+1})")
                        self.trees = self.trees[:self.best_iteration_ + 1]
                        break

        self.n_trees_fitted_ = len(self.trees)

    def _compute_ndcg(self, y, F, group):
        """Compute average NDCG across groups."""
        n_groups = len(group)
        offsets = np.zeros(n_groups + 1, dtype=np.int32)
        for i in range(n_groups):
            offsets[i+1] = offsets[i] + group[i]
            
        total_ndcg = 0.0
        valid_groups = 0
        for g in range(n_groups):
            start = offsets[g]
            end = offsets[g+1]
            y_loc = y[start:end]
            F_loc = F[start:end]
            if len(y_loc) < 2:
                continue
                
            idx = np.argsort(-F_loc)
            ideal_idx = np.argsort(-y_loc)
            
            dcg = 0.0
            idcg = 0.0
            for i in range(len(y_loc)):
                rank = i + 1
                
                rel = y_loc[idx[i]]
                dcg += ((1 << int(rel)) - 1.0) / np.log2(rank + 1.0)
                
                ideal_rel = y_loc[ideal_idx[i]]
                idcg += ((1 << int(ideal_rel)) - 1.0) / np.log2(rank + 1.0)
                
            if idcg > 0:
                total_ndcg += dcg / idcg
                valid_groups += 1
                
        return total_ndcg / valid_groups if valid_groups > 0 else 0.0

    def _fit_lambdarank(self, X_bin, y, group, X_val_bin, y_val, group_val, rng):
        """LambdaRank training loop (1 tree/round)."""
        n_samples = len(y)
        n_features = X_bin.shape[1]
        
        self.init_score_ = 0.0
        F_train = np.zeros(n_samples, dtype=np.float64)
        F_val = np.zeros(len(y_val), dtype=np.float64) if y_val is not None else None

        n_sub_samples = max(1, int(n_samples * self.subsample))
        n_sub_features = max(1, int(n_features * self.colsample_bytree))

        best_val_ndcg = -np.inf
        rounds_no_improve = 0
        all_sample_indices = np.arange(n_samples, dtype=np.int64)
        all_feature_indices = np.arange(n_features, dtype=np.int32)

        for m in range(self.n_estimators):
            grad, hess = _compute_lambdarank_gradients(y, F_train, group)

            if self.sampling_method == 'goss':
                sample_idx, feat_idx, grad_weight = self._get_goss_indices(
                    rng, grad, all_feature_indices, n_sub_features
                )
                grad *= grad_weight
                hess *= grad_weight
            else:
                sample_idx, feat_idx = self._get_subsample_indices(
                    rng, all_sample_indices, all_feature_indices,
                    n_sub_samples, n_sub_features
                )

            tf, tt, tv, tl, tr, nc, sg = self._build_one_tree(
                X_bin, grad, hess, sample_idx, feat_idx, n_features
            )
            self.trees.append((tf, tt, tv, tl, tr, nc, sg))

            tree_pred = self._predict_tree(X_bin, tf, tt, tv, tl, tr, nc)
            F_train += self.learning_rate * tree_pred

            train_ndcg = self._compute_ndcg(y, F_train, group)
            self.train_loss_history_.append(train_ndcg)

            val_ndcg = None
            if X_val_bin is not None and group_val is not None:
                val_pred = self._predict_tree(X_val_bin, tf, tt, tv, tl, tr, nc)
                F_val += self.learning_rate * val_pred
                val_ndcg = self._compute_ndcg(y_val, F_val, group_val)
                self.val_loss_history_.append(val_ndcg)

            if self.verbose and ((m + 1) % 20 == 0 or m == 0):
                msg = f"  ✓ Tree {m+1:>3}/{self.n_estimators}  |  train-NDCG: {train_ndcg:.6f}"
                if val_ndcg is not None:
                    msg += f"  |  val-NDCG: {val_ndcg:.6f}"
                print(msg)

            if self.early_stopping_rounds is not None and val_ndcg is not None:
                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    self.best_iteration_ = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"\n  ⏹ Early stopping at round {m+1} (best: {self.best_iteration_+1})")
                        self.trees = self.trees[:self.best_iteration_ + 1]
                        break

        self.n_trees_fitted_ = len(self.trees)

    # ─────────────────────────────────────────────────────────────────────
    # BINARY CLASSIFICATION TRAINING
    # ─────────────────────────────────────────────────────────────────────
    def _fit_binary(self, X_bin, y, X_val_bin, y_val, rng):
        """Binary classification training loop (sigmoid, 1 tree/round)."""
        n_samples = len(y)
        n_features = X_bin.shape[1]
        # v5.0: Use weighted mean for base_score when class weights are active
        sample_weight_init = self._compute_class_weights(y, n_samples)
        weighted_mean = np.average(y, weights=sample_weight_init)
        p0 = np.clip(weighted_mean, 1e-7, 1.0 - 1e-7)
        self.base_score_ = np.log(p0 / (1.0 - p0))
        F_train = np.full(n_samples, self.base_score_)
        F_val = np.full(len(y_val), self.base_score_) if y_val is not None else None

        n_sub_samples = max(1, int(n_samples * self.subsample))
        n_sub_features = max(1, int(n_features * self.colsample_bytree))

        sample_weight = self._compute_class_weights(y, n_samples)

        best_val_loss = np.inf
        rounds_no_improve = 0
        all_sample_indices = np.arange(n_samples, dtype=np.int64)
        all_feature_indices = np.arange(n_features, dtype=np.int32)

        for m in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F_train, -500, 500)))
            
            if self.objective == 'lambdarank':
                if getattr(self, '_group', None) is None:
                    raise ValueError("group must be provided for lambdarank objective.")
                grad, hess = _compute_lambdarank_gradients(y.astype(np.float64), F_train, np.asarray(self._group, dtype=np.int32), sigma=1.0)
            elif self.objective == 'binary_focal':
                pt = np.where(y == 1, p, 1.0 - p)
                focal_weight = (1.0 - pt) ** self.focal_gamma
                grad = (sample_weight * focal_weight * (p - y)).astype(np.float64)
                hess = (sample_weight * focal_weight * p * (1.0 - p)).astype(np.float64)
            else:
                grad = (sample_weight * (p - y)).astype(np.float64)
                hess = (sample_weight * p * (1.0 - p)).astype(np.float64)

            if self.sampling_method == 'goss':
                sample_idx, feat_idx, grad_weight = self._get_goss_indices(
                    rng, grad, all_feature_indices, n_sub_features
                )
                grad *= grad_weight
                hess *= grad_weight
            else:
                sample_idx, feat_idx = self._get_subsample_indices(
                    rng, all_sample_indices, all_feature_indices,
                    n_sub_samples, n_sub_features
                )

            tf, tt, tv, tl, tr, nc, sg = self._build_one_tree(
                X_bin, grad, hess, sample_idx, feat_idx, n_features
            )
            self.trees.append((tf, tt, tv, tl, tr, nc, sg))

            tree_pred = self._predict_tree(X_bin, tf, tt, tv, tl, tr, nc)
            F_train += self.learning_rate * tree_pred

            if self.objective == 'lambdarank':
                if self.verbose > 0:
                    train_loss = -HybridHistGBMNumbaV2._compute_ndcg(y, F_train, getattr(self, '_group', None))
                else:
                    train_loss = 0.0
            else:
                train_loss = self._compute_logloss(y, F_train)
            self.train_loss_history_.append(train_loss)

            val_loss = None
            if X_val_bin is not None:
                val_pred = self._predict_tree(X_val_bin, tf, tt, tv, tl, tr, nc)
                F_val += self.learning_rate * val_pred
                if self.objective == 'lambdarank':
                    val_loss = -HybridHistGBMNumbaV2._compute_ndcg(y_val, F_val, getattr(self, '_group_val', None))
                else:
                    val_loss = self._compute_logloss(y_val, F_val)
                self.val_loss_history_.append(val_loss)

            if self.verbose and ((m + 1) % 20 == 0 or m == 0):
                msg = f"  ✓ Tree {m+1:>3}/{self.n_estimators}  |  train-loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f"  |  val-loss: {val_loss:.6f}"
                print(msg)

            if self.early_stopping_rounds is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_iteration_ = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"\n  ⏹ Early stopping at tree {m+1} (best: {self.best_iteration_+1})")
                        self.trees = self.trees[:self.best_iteration_ + 1]
                        break

        self.n_trees_fitted_ = len(self.trees)

    # ─────────────────────────────────────────────────────────────────────
    # MULTI-CLASS CLASSIFICATION TRAINING
    # ─────────────────────────────────────────────────────────────────────
    def _fit_multiclass(self, X_bin, y, X_val_bin, y_val, rng):
        """Multi-class training loop (softmax, K trees per round)."""
        n_samples = len(y)
        n_features = X_bin.shape[1]
        K = self.n_classes_

        # ── v2.7: Vectorized one-hot encoding (no Python loop) ──
        y_onehot = np.eye(K, dtype=np.float64)[y]

        # Raw scores: (n_samples, K)
        F_train = np.zeros((n_samples, K), dtype=np.float64)
        F_val = None
        y_val_onehot = None
        if y_val is not None:
            F_val = np.zeros((len(y_val), K), dtype=np.float64)
            y_val_onehot = np.eye(K, dtype=np.float64)[y_val]

        n_sub_samples = max(1, int(n_samples * self.subsample))
        n_sub_features = max(1, int(n_features * self.colsample_bytree))

        sample_weight = self._compute_class_weights(y, n_samples)

        # Initialize per-class tree storage
        self.mc_trees = [[] for _ in range(K)]
        
        best_val_loss = np.inf
        rounds_no_improve = 0

        all_sample_indices = np.arange(n_samples, dtype=np.int64)
        all_feature_indices = np.arange(n_features, dtype=np.int32)

        for m in range(self.n_estimators):
            # Compute softmax probabilities
            proba = self._softmax(F_train)  # (n_samples, K)

            if self.sampling_method == 'goss':
                # For multi-class, we use the max absolute gradient across classes to sample
                max_abs_grad = np.max(np.abs(proba - y_onehot), axis=1) * sample_weight
                sample_idx, feat_idx, grad_weight = self._get_goss_indices(
                    rng, max_abs_grad, all_feature_indices, n_sub_features
                )
            else:
                # Shared subsampling for all K trees in this round
                sample_idx, feat_idx = self._get_subsample_indices(
                    rng, all_sample_indices, all_feature_indices,
                    n_sub_samples, n_sub_features
                )
                grad_weight = None

            # ── v2.7: Vectorized gradient/hessian — single numpy op for all K ──
            grad_all = sample_weight[:, None] * (proba - y_onehot)   # (n, K)
            hess_all = sample_weight[:, None] * proba * (1.0 - proba) # (n, K)

            # Build K trees sequentially
            for k in range(K):
                grad_k = np.ascontiguousarray(grad_all[:, k])
                hess_k = np.maximum(np.ascontiguousarray(hess_all[:, k]), 1e-8)
                
                if grad_weight is not None:
                    grad_k *= grad_weight
                    hess_k *= grad_weight

                tf, tt, tv, tl, tr, nc, sg = self._build_one_tree(
                    X_bin, grad_k, hess_k, sample_idx, feat_idx, n_features
                )
                self.mc_trees[k].append((tf, tt, tv, tl, tr, nc, sg))
                
                tree_pred = self._predict_tree(X_bin, tf, tt, tv, tl, tr, nc)
                F_train[:, k] += self.learning_rate * tree_pred
                
                if F_val is not None and X_val_bin is not None:
                    val_pred = self._predict_tree(X_val_bin, tf, tt, tv, tl, tr, nc)
                    F_val[:, k] += self.learning_rate * val_pred

            train_loss = self._compute_multiclass_logloss(y_onehot, F_train)
            self.train_loss_history_.append(train_loss)

            val_loss = None
            if F_val is not None:
                val_loss = self._compute_multiclass_logloss(y_val_onehot, F_val)
                self.val_loss_history_.append(val_loss)

            if self.verbose and ((m + 1) % 20 == 0 or m == 0):
                msg = f"  ✓ Round {m+1:>3}/{self.n_estimators}  |  {K} trees  |  train-loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f"  |  val-loss: {val_loss:.6f}"
                print(msg)
                
            if self.early_stopping_rounds is not None and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_iteration_ = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"\n  ⏹ Early stopping at round {m+1} (best: {self.best_iteration_+1})")
                        for k in range(K):
                            self.mc_trees[k] = self.mc_trees[k][:self.best_iteration_ + 1]
                        break

        self.n_trees_fitted_ = sum(len(t) for t in self.mc_trees)

    # ─────────────────────────────────────────────────────────────────────
    # MAIN FIT
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, X, y, group=None, X_val=None, y_val=None, group_val=None, eval_set=None):
        """Train with auto-detection of binary vs multi-class.
        
        v3.0: Supports eval_set=[(X_val, y_val)] for validation-based early stopping,
        native NaN handling, monotonic constraints, and categorical features.
        
        Parameters
        ----------
        X : np.ndarray
            Training features. May contain NaN values (handled natively).
        y : np.ndarray
            Training labels.
        X_val, y_val : np.ndarray or None
            Validation data (legacy API). Prefer eval_set instead.
        eval_set : list of tuples or None
            Validation set as [(X_val, y_val)]. If provided, overrides X_val/y_val.
            Used for validation-based early stopping.
        """
        self._group = group
        self._group_val = group_val
        if self.max_depth <= 0 and self.max_depth != 'auto':
            raise ValueError(f"max_depth must be > 0, got {self.max_depth}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {self.n_estimators}")

        # v3.0: Support eval_set=[(X_val, y_val)] syntax (like XGBoost/LightGBM)
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]

        # ─── Detect number of classes ───
        if self.objective == 'regression':
            self._is_multiclass = False
            self.classes_ = None
            self.n_classes_ = 1
            y = np.array(y, dtype=np.float64)
            if y_val is not None:
                y_val = np.array(y_val, dtype=np.float64)
        elif self.objective == 'lambdarank':
            self._is_multiclass = False
            self.classes_ = None
            self.n_classes_ = 1
            y = np.array(y, dtype=np.float64)
            if y_val is not None:
                y_val = np.array(y_val, dtype=np.float64)
        else:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self._is_multiclass = self.n_classes_ > 2

            # Remap labels to 0..K-1 for all modes
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([label_map[v] for v in y], dtype=np.int64)
            if y_val is not None:
                y_val = np.array([label_map[v] for v in y_val], dtype=np.int64)

        if self.objective == 'regression':
            mode_str = "Regression (Continuous)"
        elif self.objective == 'lambdarank':
            mode_str = "Ranking (LambdaMART)"
        elif self._is_multiclass:
            mode_str = f"Multi-class ({self.n_classes_} classes)"
        else:
            mode_str = "Binary"
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"  Training Hybrid GBM v3.0 — {mode_str}")
            print("=" * 70)

        # ─── Categorical Encoding ───
        if self.categorical_features is not None:
            if self.verbose:
                print(f"  📌 Detected {len(self.categorical_features)} categorical features. Encoding top 63 classes...")
            X = self._encode_categorical(X, is_fit=True)
            if X_val is not None:
                X_val = self._encode_categorical(X_val, is_fit=False)

        # ─── Build Histogram Bin Edges ───
        self.trees = []
        self.mc_trees = None
        self.train_loss_history_ = []
        self.val_loss_history_ = []
        self.n_features_in_ = X.shape[1]

        rng = np.random.RandomState(self.random_state)
        self._auto_no_weights = False
        self._auto_full_weights = False

        # v3.0: Initialize monotonic constraints
        if self.monotone_constraints is not None:
            mc = self.monotone_constraints
            # v5.0: Support dict format {feature_idx: direction}
            if isinstance(mc, dict):
                mc_arr = np.zeros(X.shape[1], dtype=np.int32)
                for feat_idx, direction in mc.items():
                    mc_arr[feat_idx] = direction
                self._mono_arr = mc_arr
            else:
                self._mono_arr = np.array(mc, dtype=np.int32)
            if len(self._mono_arr) != X.shape[1]:
                raise ValueError(
                    f"monotone_constraints length ({len(self._mono_arr)}) must match "
                    f"n_features ({X.shape[1]})"
                )
            n_constrained = np.sum(self._mono_arr != 0)
            if self.verbose and n_constrained > 0:
                print(f"  🔒 Monotonic constraints: {n_constrained} features constrained")
        else:
            self._mono_arr = None

        # v3.0: Initialize categorical features
        if self.categorical_features is not None:
            self._categorical_set = set(self.categorical_features)
            self._is_categorical = np.zeros(X.shape[1], dtype=bool)
            for idx in self.categorical_features:
                self._is_categorical[idx] = True
            if self.verbose:
                print(f"  📋 Categorical features: {len(self._categorical_set)} columns ({list(self._categorical_set)[:5]}{'...' if len(self._categorical_set) > 5 else ''})")
        else:
            self._categorical_set = set()
            self._is_categorical = np.zeros(X.shape[1], dtype=bool)

        # Convert X to float64 if needed (handles mixed types and float16)
        if not np.issubdtype(X.dtype, np.floating) or X.dtype == np.float16:
            X = X.astype(np.float64)
        if X_val is not None and (not np.issubdtype(X_val.dtype, np.floating) or X_val.dtype == np.float16):
            X_val = X_val.astype(np.float64)

        # v3.0: Log NaN detection
        n_nan_features = np.sum(np.any(np.isnan(X), axis=0)) if np.any(np.isnan(X)) else 0
        if self.verbose and n_nan_features > 0:
            print(f"  🕳️  Native NaN handling: {n_nan_features} features with missing values")

        self._compute_adaptive_hyperparameters(X, y)

        sample_weight = None
        if getattr(self, 'scale_pos_weight', None) is not None and self.objective in ('binary', 'binary_focal'):
            sample_weight = np.ones(len(y), dtype=np.float64)
            sample_weight[y == 1] = self.scale_pos_weight
            
        self.bin_edges, self.feature_bin_counts = self._build_bin_edges(X, sample_weight=sample_weight)
        self._max_bins = int(np.max(self.feature_bin_counts))
        X_bin, bin_time = self._bin_features(X)
        self.times['binning'] = bin_time

        X_val_bin = None
        if X_val is not None and y_val is not None:
            X_val_bin, _ = self._bin_features(X_val)

        self._warmup_jit(X_bin)

        t0_train = time.perf_counter()

        # ─── Route to binary, multi-class, or regression training ───
        if self.objective == 'regression':
            self._fit_regression(X_bin, y, X_val_bin, y_val, rng)
        elif self.objective == 'lambdarank':
            if self._group is None:
                raise ValueError("group parameter is strictly required for LambdaRank objective.")
            self._fit_lambdarank(X_bin, y, self._group, X_val_bin, y_val, self._group_val, rng)
        elif self._is_multiclass:
            self._fit_multiclass(X_bin, y, X_val_bin, y_val, rng)
        else:
            self._fit_binary(X_bin, y, X_val_bin, y_val, rng)

        self.times['training'] = (time.perf_counter() - t0_train) * 1000

        self._pack_trees()
        self._compute_feature_importances()

        total_trees = self.n_trees_fitted_
        if self.verbose:
            print(f"\n  ✓ Training complete! ({total_trees} trees)")
            print(f"    Binning:  {self.times['binning']:.2f} ms")
            print(f"    Training: {self.times['training']:.2f} ms")
            print(f"    Total:    {self.times['binning'] + self.times['training']:.2f} ms")
            if self.early_stopping_rounds is not None and X_val_bin is not None:
                if self.early_stopping_rounds is not None and X_val_bin is not None:
                    print(f"    Best iteration: {self.best_iteration_ + 1}")

        return self


    def predict_proba(self, X):
        if getattr(self, 'n_features_in_', None) is None:
            raise ValueError("This HybridHistGBMNumbaV2 instance is not fitted yet.")
            
        if self.categorical_features is not None:
            X = self._encode_categorical(X, is_fit=False)
            
        if not np.issubdtype(X.dtype, np.floating) or X.dtype == np.float16:
            X = X.astype(np.float64)
        X_bin, _ = self._bin_features(X)
        t0 = time.perf_counter()

        if self._is_multiclass:
            # ─── Multi-class: predict K sets of trees, then softmax ───
            K = self.n_classes_
            n = len(X)
            F = np.zeros((n, K), dtype=np.float64)
            for k in range(K):
                if HAS_NUMBA and self._mc_packed is not None:
                    pk = self._mc_packed[k]
                    F[:, k] = predict_all_trees_jit(
                        X_bin, pk['features'], pk['thresholds'],
                        pk['values'], pk['left'], pk['right'],
                        pk['offsets'], np.int32(len(self.mc_trees[k])),
                        np.float64(self.learning_rate), self._get_is_categorical(X_bin.shape[1])
                    )
                else:
                    for tf, tt, tv, tl, tr, nc, _ in self.mc_trees[k]:
                        for i in range(n):
                            node = 0
                            while tf[node] >= 0:
                                if X_bin[i, tf[node]] <= tt[node]: node = tl[node]
                                else: node = tr[node]
                            F[i, k] += self.learning_rate * tv[node]
            self.times['inference'] = (time.perf_counter() - t0) * 1000
            return self._softmax(F)
        else:
            # ─── Binary: existing fast path ───
            if HAS_NUMBA and self._packed_features is not None:
                F = predict_all_trees_jit(
                    X_bin, self._packed_features, self._packed_thresholds,
                    self._packed_values, self._packed_left, self._packed_right,
                    self._tree_offsets, np.int32(len(self.trees)),
                    np.float64(self.learning_rate), self._get_is_categorical(X_bin.shape[1])
                ) + getattr(self, 'base_score_', 0.0)
            else:
                F = np.full(len(X), getattr(self, 'base_score_', 0.0))
                for tf, tt, tv, tl, tr, nc, _ in self.trees:
                    for i in range(len(X)):
                        node = 0
                        while tf[node] >= 0:
                            if X_bin[i, tf[node]] <= tt[node]: node = tl[node]
                            else: node = tr[node]
                        F[i] += self.learning_rate * tv[node]
            self.times['inference'] = (time.perf_counter() - t0) * 1000
            proba = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            return np.column_stack([1 - proba, proba])

    def _get_is_categorical(self, n_features):
        is_categorical = np.zeros(n_features, dtype=np.bool_)
        if self.categorical_features is not None:
            for j in self.categorical_features:
                is_categorical[j] = True
        return is_categorical

    def predict(self, X):
        if self.objective in ('regression', 'lambdarank'):
            X_bin, _ = self._bin_features(X)
            preds = np.full(X.shape[0], self.init_score_, dtype=np.float64)
            if HAS_NUMBA and self._packed_features is not None:
                preds += predict_all_trees_jit(
                    X_bin, self._packed_features, self._packed_thresholds,
                    self._packed_values, self._packed_left, self._packed_right,
                    self._tree_offsets, np.int32(len(self.trees)), np.float64(self.learning_rate), self._get_is_categorical(X_bin.shape[1])
                )
            return preds
            
        proba = self.predict_proba(X)
        if self._is_multiclass:
            # Return original class labels, not 0..K-1
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            return self.classes_[(proba[:, 1] > 0.5).astype(int)]

    def predict_contributions(self, X):
        """
        Computes native TreeSHAP (Saabas heuristic) feature contributions at C-speed.
        Returns a matrix of shape (n_samples, n_features + 1), where the last column 
        is the expected value (bias) of the model.
        For multi-class, returns a list of such matrices (one per class).
        """
        if not HAS_NUMBA:
            raise NotImplementedError("Native SHAP requires Numba.")
            
        X_bin, _ = self._bin_features(X)
        n_features = X_bin.shape[1]
        is_cat = self._get_is_categorical(n_features)
        
        if self.objective in ('regression', 'lambdarank'):
            if self._packed_features is not None:
                contribs = predict_contributions_all_trees_jit(
                    X_bin, self._packed_features, self._packed_thresholds,
                    self._packed_values, self._packed_left, self._packed_right,
                    self._tree_offsets, np.int32(len(self.trees)), np.float64(self.learning_rate), is_cat, np.int32(n_features)
                )
                contribs[:, -1] += self.init_score_
                return contribs
        else:
            # Classification
            if not self._is_multiclass:
                if self._packed_features is not None:
                    contribs = predict_contributions_all_trees_jit(
                        X_bin, self._packed_features, self._packed_thresholds,
                        self._packed_values, self._packed_left, self._packed_right,
                        self._tree_offsets, np.int32(len(self.trees)), np.float64(self.learning_rate), is_cat, np.int32(n_features)
                    )
                    contribs[:, -1] += self.init_score_
                    return contribs
            else:
                # Multi-class
                all_contribs = []
                for k in range(self.n_classes_):
                    if self._mc_packed is not None and len(self._mc_packed) > k:
                        pk = self._mc_packed[k]
                        if pk is not None:
                            contribs = predict_contributions_all_trees_jit(
                                X_bin, pk['features'], pk['thresholds'],
                                pk['values'], pk['left'], pk['right'],
                                pk['offsets'], np.int32(len(self.mc_trees[k])), np.float64(self.learning_rate), is_cat, np.int32(n_features)
                            )
                            contribs[:, -1] += self.init_score_
                            all_contribs.append(contribs)
                return all_contribs
        return None

    def export_to_onnx(self, filepath, input_name="X"):
        """
        Exports the model to ONNX format (TreeEnsembleClassifier / Regressor).
        Allows ultra-low latency inference in C++, Java, Rust, or C#.
        """
        try:
            import onnx
            from onnx import helper
            from onnx import TensorProto
        except ImportError:
            raise ImportError("ONNX export requires the 'onnx' package. Run `pip install onnx`.")
            
        if self.categorical_features is not None and len(self.categorical_features) > 0:
            raise NotImplementedError("ONNX export currently does not support native categorical bitset splits. Please use numerical features or OHE.")
            
        nodes_treeids = []
        nodes_nodeids = []
        nodes_featureids = []
        nodes_values = []
        nodes_modes = []
        nodes_truenodeids = []
        nodes_falsenodeids = []
        nodes_missing_value_tracks_true = []
        
        class_treeids = []
        class_nodeids = []
        class_ids = []
        class_weights = []

        is_clf = self.objective not in ('regression', 'lambdarank')
        
        def traverse_tree(k, t, offset, node_id, is_clf, lr):
            nodes_treeids.append(t)
            nodes_nodeids.append(node_id)
            
            feat = self._packed_features[offset + node_id] if not is_clf and not self._is_multiclass else (self._mc_packed[k]['features'][offset + node_id] if self._is_multiclass else self._packed_features[offset + node_id])
            
            if feat < 0:
                # Leaf
                nodes_modes.append(b"LEAF")
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_value_tracks_true.append(0)
                
                val = self._packed_values[offset + node_id] if not is_clf and not self._is_multiclass else (self._mc_packed[k]['values'][offset + node_id] if self._is_multiclass else self._packed_values[offset + node_id])
                
                class_treeids.append(t)
                class_nodeids.append(node_id)
                class_ids.append(k if is_clf and self._is_multiclass else 0)
                class_weights.append(float(val * lr))
            else:
                # Branch
                nodes_modes.append(b"BRANCH_LEQ")
                nodes_featureids.append(int(feat))
                
                bin_idx = int(self._packed_thresholds[offset + node_id] if not is_clf and not self._is_multiclass else (self._mc_packed[k]['thresholds'][offset + node_id] if self._is_multiclass else self._packed_thresholds[offset + node_id]))
                
                if self.categorical_features is not None and int(feat) in self.categorical_features:
                    # Categorical feature (unsupported by ONNX TreeEnsemble directly, fallback to float)
                    thresh = float(bin_idx)
                else:
                    edges = self.bin_edges[int(feat)]
                    if bin_idx + 1 < len(edges):
                        thresh = float(edges[bin_idx + 1])
                    else:
                        thresh = float(edges[-1])
                
                nodes_values.append(thresh)
                
                left_id = self._packed_left[offset + node_id] if not is_clf and not self._is_multiclass else (self._mc_packed[k]['left'][offset + node_id] if self._is_multiclass else self._packed_left[offset + node_id])
                right_id = self._packed_right[offset + node_id] if not is_clf and not self._is_multiclass else (self._mc_packed[k]['right'][offset + node_id] if self._is_multiclass else self._packed_right[offset + node_id])
                
                nodes_truenodeids.append(int(left_id))
                nodes_falsenodeids.append(int(right_id))
                nodes_missing_value_tracks_true.append(0)
                
                traverse_tree(k, t, offset, left_id, is_clf, lr)
                traverse_tree(k, t, offset, right_id, is_clf, lr)

        lr = self.learning_rate
        
        if is_clf and self._is_multiclass:
            t_idx = 0
            for k in range(self.n_classes_):
                pk = self._mc_packed[k]
                for m in range(len(self.mc_trees[k])):
                    traverse_tree(k, t_idx, pk['offsets'][m], 0, is_clf, lr)
                    t_idx += 1
        else:
            for t in range(len(self.trees)):
                traverse_tree(0, t, self._tree_offsets[t], 0, is_clf, lr)

        # Build ONNX node
        attr = {
            "nodes_treeids": nodes_treeids,
            "nodes_nodeids": nodes_nodeids,
            "nodes_featureids": nodes_featureids,
            "nodes_values": nodes_values,
            "nodes_modes": nodes_modes,
            "nodes_truenodeids": nodes_truenodeids,
            "nodes_falsenodeids": nodes_falsenodeids,
            "nodes_missing_value_tracks_true": nodes_missing_value_tracks_true,
            "base_values": [float(self.init_score_)] if not (is_clf and self._is_multiclass) else [float(self.init_score_)] * self.n_classes_
        }
        
        if is_clf:
            attr["class_treeids"] = class_treeids
            attr["class_nodeids"] = class_nodeids
            attr["class_ids"] = class_ids
            attr["class_weights"] = class_weights
            attr["classlabels_int64s"] = list(range(self.n_classes_))
            attr["post_transform"] = b"SOFTMAX" if self._is_multiclass else b"LOGISTIC"
            node_op = "TreeEnsembleClassifier"
            output_names = ["label", "probabilities"]
        else:
            attr["target_treeids"] = class_treeids
            attr["target_nodeids"] = class_nodeids
            attr["target_ids"] = class_ids
            attr["target_weights"] = class_weights
            attr["n_targets"] = 1
            node_op = "TreeEnsembleRegressor"
            output_names = ["prediction"]

        tree_node = helper.make_node(
            node_op,
            inputs=[input_name],
            outputs=output_names,
            name="BSSCL_GBM_TreeEnsemble",
            domain="ai.onnx.ml",
            **attr
        )
        
        n_features = len(self.feature_importances_)
        X_in = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, n_features])
        
        if is_clf:
            Y_label = helper.make_tensor_value_info("label", TensorProto.INT64, [None])
            Y_prob = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, self.n_classes_])
            graph = helper.make_graph([tree_node], "BSSCL_GBM", [X_in], [Y_label, Y_prob])
        else:
            Y_pred = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [None, 1])
            graph = helper.make_graph([tree_node], "BSSCL_GBM", [X_in], [Y_pred])
            
        opsets = [helper.make_opsetid("", 14), helper.make_opsetid("ai.onnx.ml", 3)]
        model = helper.make_model(graph, producer_name="bsscl-gbm-v1.2.0", opset_imports=opsets)
        import onnx
        onnx.save(model, filepath)
        print(f"✅ ONNX model successfully exported to {filepath}")

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


    def plot_feature_importance(self, importance_type='gain', top_n=10, feature_names=None):
        """Plots the feature importances using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Please install matplotlib to use plot_feature_importance().")
            
        if getattr(self, 'feature_importances_', None) is None:
            raise ValueError("Model must be fitted before plotting feature importance.")
            
        if importance_type == 'gain':
            importances = self.feature_importances_
        elif importance_type == 'split':
            importances = self.feature_split_counts_
        else:
            raise ValueError("importance_type must be 'gain' or 'split'.")
            
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
        # Sort features by importance
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances ({importance_type.capitalize()})")
        plt.barh(range(len(indices)), importances[indices], color='#2c3e50', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel(f"Relative {importance_type.capitalize()}")
        plt.tight_layout()
        plt.show()

    def get_shap_explainer(self, background_data=None, use_binned_thresholds=False):
        """
        Constructs and returns a shap.TreeExplainer for this model.
        Supports binary, multi-class, and regression.
        
        Args:
            background_data: The background dataset (X_train or X_test) used to compute expected values.
            use_binned_thresholds: If True, uses exact internal bin indices for perfectly accurate 0.0000 
                                   additivity SHAP values. You must pass binned data to shap_values().
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Please install shap (pip install shap) to use get_shap_explainer().")
            
        if getattr(self, 'n_features_in_', None) is None:
            raise ValueError("Model must be fitted before generating SHAP explainer.")
            
        trees_list = []
        
        # Multi-class models are built as K trees per round. SHAP expects them flat, but with objective setup.
        # Actually, SHAP's custom format expects trees to be a flat list, and values shape (nodes, K).
        # But BSSCL-GBM builds K independent trees per round, each outputting a scalar.
        # So we can just format it like LightGBM: flat list of trees, and objective='multiclass' if multiclass.
        
        if self._is_multiclass:
            # SHAP expects flat list where tree 0 is class 0, tree 1 is class 1...
            flat_trees = []
            n_rounds = len(self.mc_trees[0])
            K = self.n_classes_
            for m in range(n_rounds):
                for k in range(K):
                    if m < len(self.mc_trees[k]):
                        flat_trees.append(self.mc_trees[k][m])
        else:
            flat_trees = self.trees
            K = 1
            
        for tree_index, tree_data in enumerate(flat_trees):
            tf, tt, tv, tl, tr, nc, sg = tree_data
            
            # tf is -1 for leaves, SHAP uses -1
            # tl, tr is -1 for leaves, SHAP uses -1
            
            # BSSCL-GBM builds nodes in order up to nc
            left = tl[:nc].copy()
            right = tr[:nc].copy()
            features = tf[:nc].copy()
            thresholds_bin = tt[:nc].copy()
            values = tv[:nc].copy()
            
            # Map bin thresholds back to original float thresholds (unless exact mode is requested)
            if use_binned_thresholds:
                thresholds_final = thresholds_bin.copy()
            else:
                thresholds_final = np.zeros(nc, dtype=np.float64)
                for i in range(nc):
                    if features[i] >= 0:
                        feat = features[i]
                        bin_idx = int(thresholds_bin[i])
                        edges = self.bin_edges[feat]
                        if bin_idx + 1 < len(edges):
                            thresholds_final[i] = edges[bin_idx + 1]
                        else:
                            thresholds_final[i] = edges[-1]
                    else:
                        thresholds_final[i] = -1.0
            
            # Make weights and values mathematically consistent via a bottom-up pass
            # This allows SHAP to instantly compute expected values in <10ms without background data!
            weights = np.zeros(nc, dtype=np.float64)
            values_consistent = np.zeros(nc, dtype=np.float64)
            
            for i in range(nc - 1, -1, -1):
                if features[i] < 0: # Leaf
                    weights[i] = 1.0
                    values_consistent[i] = values[i] * self.learning_rate
                else: # Internal node
                    L = left[i]
                    R = right[i]
                    weights[i] = weights[L] + weights[R]
                    values_consistent[i] = (values_consistent[L] * weights[L] + values_consistent[R] * weights[R]) / weights[i]
            
            # Format values for SHAP: (nc, K)
            values_multiclass = np.zeros((nc, K), dtype=np.float64)
            if self._is_multiclass:
                class_id = tree_index % K
                values_multiclass[:, class_id] = values_consistent
            else:
                values_multiclass[:, 0] = values_consistent
            
            tree_dict = {
                "children_left": left,
                "children_right": right,
                "children_default": left, # Default to left for NaNs
                "features": features,
                "thresholds": thresholds_final,
                "values": values_multiclass,
                "node_sample_weight": weights
            }
            trees_list.append(tree_dict)
            
        if self.objective == 'regression':
            objective = "squared_error" if self.loss == 'mse' else "absolute_error"
        elif self._is_multiclass:
            objective = "multiclass"
        else:
            objective = "binary_crossentropy"
            
        base_offset = getattr(self, 'init_score_', 0.0)
        if self._is_multiclass:
            # Multi-class base offset is usually an array
            base_offset = np.zeros(self.n_classes_)
            
        model_dict = {
            "trees": trees_list,
            "objective": objective,
            "tree_output": "raw_value",
            "base_offset": base_offset
        }
        
        if background_data is not None:
            return shap.TreeExplainer(model_dict, data=background_data, feature_perturbation="interventional")
        return shap.TreeExplainer(model_dict)

    def summary(self):
        if self.n_trees_fitted_ == 0:
            print("  Model not fitted."); return
        if self.objective == 'regression':
            mode_str = "Regression (Continuous)"
        elif self.objective == 'lambdarank':
            mode_str = "Ranking (LambdaMART)"
        elif self._is_multiclass:
            mode_str = f"Multi-class ({self.n_classes_} classes)"
        else:
            mode_str = "Binary"
        print("\n" + "=" * 60)
        print(f"  Hybrid GBM v3.0 — Model Summary ({mode_str})")
        print("=" * 60)
        print(f"  Classes:       {self.n_classes_} {list(self.classes_) if len(self.classes_) <= 10 else '...'}")
        print(f"  Total trees:   {self.n_trees_fitted_}")
        if self._is_multiclass:
            print(f"  Trees/class:   {len(self.mc_trees[0])} × {self.n_classes_} classes")
        print(f"  Max depth:     {self.max_depth}")
        print(f"  Learning rate: {self.learning_rate:.4f}")
        print(f"  Subsample:     {self.subsample}")
        print(f"  Colsample:     {self.colsample_bytree}")
        print(f"  Lambda (L2):   {self.reg_lambda}")
        print(f"  Gamma:         {self.gamma}")
        print(f"  Final loss:    {self.train_loss_history_[-1]:.6f}")
        print(f"  Train time:    {self.times['training']:.2f} ms")
        print(f"  Numba JIT:     {'✓' if HAS_NUMBA else '✗'}")
        if self.feature_importances_ is not None:
            top_k = min(5, len(self.feature_importances_))
            top_idx = np.argsort(self.feature_importances_)[::-1][:top_k]
            print(f"\n  Top {top_k} Features:")
            for rank, idx in enumerate(top_idx):
                pct = self.feature_importances_[idx] * 100
                print(f"    #{rank+1}  Feature {idx:>3}  |  {pct:5.1f}% gain  |  {self.feature_split_counts_[idx]:>4} splits")
        print("=" * 60)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth_init,
            'learning_rate': self.learning_rate_init,
            'n_bins': self.n_bins_init,
            'reg_lambda': self.reg_lambda,
            'min_samples_leaf': self.min_samples_leaf,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'class_weight': self.class_weight,
            'monotone_constraints': self.monotone_constraints,
            'categorical_features': self.categorical_features,
            'early_stopping_rounds': self.early_stopping_rounds,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __repr__(self):
        return (f"HybridHistGBMNumbaV2(n_estimators={self.n_estimators}, "
                f"max_depth='{self.max_depth_init}', subsample={self.subsample}, "
                f"colsample_bytree={self.colsample_bytree})")

    # ═══════════════════════════════════════════════════════════════════════
    # PRODUCTION DEPLOYMENT SUPPORT (Multiprocessing-Safe)
    # ═══════════════════════════════════════════════════════════════════════

    def warmup(self, n_features=None):
        """
        Pre-compile all Numba JIT functions to eliminate cold-start latency.

        Call this once at application/container startup BEFORE serving
        predictions. This avoids JIT compilation delays on the first
        real request.

        Parameters
        ----------
        n_features : int or None
            Number of features in your data. If None, uses the number
            of features the model was trained on.

        Example
        -------
        >>> model = joblib.load('model.joblib')
        >>> model.warmup()  # Pre-compile JIT at startup
        >>> # Now ready to serve real-time predictions
        """
        if not HAS_NUMBA:
            return

        nf = n_features or self.n_features_in_
        if nf is None:
            nf = 10  # Fallback for unfitted models

        # Create a small dummy input to trigger JIT compilation
        dummy_X = np.random.rand(2, nf).astype(np.float64)
        try:
            # We don't care about the result, we just want to force JIT compilation
            self.predict_proba(dummy_X)
        except (ValueError, AttributeError):
            pass  # nosec B110  # Model may not be fitted yet; that's OK

    def predict_batch_parallel(self, X, n_workers=4, batch_size=10000):
        """
        Multiprocessing-safe batch prediction for large datasets.

        Uses process-based parallelism (not threads) to safely leverage
        multiple CPU cores without Numba threading conflicts.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        n_workers : int, default=4
            Number of parallel worker processes.
        batch_size : int, default=10000
            Number of rows per batch sent to each worker.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels.

        Example
        -------
        >>> # For large datasets (100k+ rows):
        >>> predictions = model.predict_batch_parallel(X_large, n_workers=4)
        """
        import math
        import multiprocessing as mp

        n = len(X)
        if n <= batch_size:
            return self.predict(X)

        n_batches = math.ceil(n / batch_size)
        batches = [X[i * batch_size: min((i + 1) * batch_size, n)] for i in range(n_batches)]

        # Use explicitly 'spawn' context to prevent Numba thread deadlocks on Linux
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(self.predict, batches)

        return np.concatenate(results)

    def predict_proba_batch_parallel(self, X, n_workers=4, batch_size=10000):
        """
        Multiprocessing-safe batch probability prediction for large datasets.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        n_workers : int, default=4
            Number of parallel worker processes.
        batch_size : int, default=10000
            Number of rows per batch sent to each worker.

        Returns
        -------
        probabilities : np.ndarray
            Predicted class probabilities.
        """
        import math
        from multiprocessing import Pool

        n = len(X)
        if n <= batch_size:
            return self.predict_proba(X)

        n_batches = math.ceil(n / batch_size)
        batches = [X[i * batch_size: min((i + 1) * batch_size, n)] for i in range(n_batches)]

        with Pool(processes=n_workers) as pool:
            results = pool.map(self.predict_proba, batches)

        return np.vstack(results)

    def production_info(self):
        """
        Display production deployment configuration and recommendations.

        Example
        -------
        >>> model = joblib.load('model.joblib')
        >>> model.production_info()
        """
        print("\n" + "=" * 60)
        print("  🚀 PRODUCTION DEPLOYMENT INFO")
        print("=" * 60)
        print("  Model:            HybridHistGBMNumbaV2")
        print(f"  Numba JIT:        {'✅ Active' if HAS_NUMBA else '❌ Not installed'}")
        print(f"  Trees fitted:     {self.n_trees_fitted_}")
        print(f"  Features:         {self.n_features_in_}")
        print(f"  Classes:          {self.n_classes_}")
        print(f"  Class weights:    {self.class_weight}")
        print("\n  ⚡ Deployment Recommendations:")
        print("  ─────────────────────────────────────────")
        print("  Web Server:       Gunicorn / Uvicorn with multiprocessing")
        print("  Workers:          gunicorn app:app --workers 4")
        print("  Parallelism:      Use multiprocessing (NOT threads)")
        print("  Warmup:           Call model.warmup() at startup")
        print("  Serialization:    joblib.dump(model, 'model.joblib')")
        print("  Batch Predict:    model.predict_batch_parallel(X, n_workers=4)")
        print("  Single-Row SLA:   < 0.2ms (P99)")
        print("  Max Throughput:   ~3.7M rows/sec")
        print("=" * 60)


class HybridHistGBMRegressor(HybridHistGBMNumbaV1_2_0):
    """
    Enterprise-Grade Histogram GBM Regressor.
    """
    def __init__(self, n_estimators='auto', max_depth='auto', learning_rate='auto',
                 n_bins='auto', reg_lambda=1.0, min_samples_leaf=5,
                 min_child_weight='auto', gamma='auto',
                 subsample='auto', colsample_bytree='auto',
                 grow_policy='leaf', max_leaves=31, binning_strategy='auto',
                 monotone_constraints=None, categorical_features=None,
                 early_stopping_rounds=None, verbose=True, random_state=None,
                 loss='mse', device='cpu'):
        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            n_bins=n_bins, reg_lambda=reg_lambda, min_samples_leaf=min_samples_leaf,
            min_child_weight=min_child_weight, gamma=gamma,
            subsample=subsample, colsample_bytree=colsample_bytree,
            class_weight=None, grow_policy=grow_policy, max_leaves=max_leaves, 
            binning_strategy=binning_strategy, monotone_constraints=monotone_constraints, 
            categorical_features=categorical_features, early_stopping_rounds=early_stopping_rounds, 
            verbose=verbose, random_state=random_state, objective='regression', focal_gamma=2.0, loss=loss, device=device
        )


class HybridHistGBMRanker(HybridHistGBMNumbaV1_2_0):
    """
    Hybrid Histogram Gradient Boosting Ranker.
    
    Implements LambdaMART (lambdarank objective) for learning-to-rank tasks.
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 n_bins='auto',
                 min_samples_leaf=20,
                 gamma=0.0,
                 reg_lambda=1.0,
                 min_child_weight=1e-5,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 grow_policy='depth',
                 max_leaves=31,
                 random_state=None,
                 binning_strategy='auto',
                 early_stopping_rounds=None,
                 monotone_constraints=None,
                 categorical_features=None,
                 verbose=1,
                 device='cpu'):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='lambdarank',
            n_bins=n_bins,
            min_samples_leaf=min_samples_leaf,
            gamma=gamma,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            grow_policy=grow_policy,
            max_leaves=max_leaves,
            random_state=random_state,
            binning_strategy=binning_strategy,
            early_stopping_rounds=early_stopping_rounds,
            monotone_constraints=monotone_constraints,
            categorical_features=categorical_features,
            verbose=verbose
        )
        
    def predict(self, X):
        """Return raw scores for ranking."""
        if getattr(self, 'n_features_in_', None) is None:
            raise ValueError("This HybridHistGBMRanker instance is not fitted yet.")
            
        if self.categorical_features is not None:
            X = self._encode_categorical(X, is_fit=False)
            
        if not np.issubdtype(X.dtype, np.floating) or X.dtype == np.float16:
            X = X.astype(np.float64)
            
        X_bin, _ = self._bin_features(X)
        preds = np.zeros(len(X), dtype=np.float64)
        if HAS_NUMBA and self._packed_features is not None:
            preds += predict_all_trees_jit(
                X_bin, self._packed_features, self._packed_thresholds,
                self._packed_values, self._packed_left, self._packed_right,
                self._tree_offsets, np.int32(len(self.trees)), np.float64(self.learning_rate), self._get_is_categorical(X_bin.shape[1])
            )
        return preds
