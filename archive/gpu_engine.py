import os
import tempfile
import time

import numpy as np

# Import base class and lambdarank gradients
from .estimator_v1_0_1 import HybridHistGBMNumbaV1_0_1 as HybridHistGBMNumbaV2, _compute_lambdarank_gradients

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    # ═══════════════════════════════════════════════════════════════════════
    # INDUSTRIAL GPU CUDA KERNELS
    # ═══════════════════════════════════════════════════════════════════════
    @cuda.jit
    def _build_histograms_single_node_cuda(X_bin, grad, hess, histograms, node_indices,
                                            row_subset, feature_subset,
                                            n_sub_samples, n_sub_features, target_node):
        i = cuda.grid(1)
        if i < n_sub_samples:
            row_id = row_subset[i]
            if node_indices[row_id] == target_node:
                g = grad[row_id]
                h = hess[row_id]
                for f_idx in range(n_sub_features):
                    j = feature_subset[f_idx]
                    bin_val = X_bin[row_id, j]
                    cuda.atomic.add(histograms, (j, bin_val, 0), g)
                    cuda.atomic.add(histograms, (j, bin_val, 1), h)

    @cuda.jit
    def _find_best_split_from_hist_cuda(histograms, split_results, feature_subset,
                                         n_sub_features, max_bins, reg_lambda,
                                         gamma, min_child_weight):
        idx = cuda.grid(1)
        if idx < n_sub_features:
            j = feature_subset[idx]
            sum_g, sum_h = 0.0, 0.0
            for b in range(max_bins):
                sum_g += histograms[j, b, 0]
                sum_h += histograms[j, b, 1]
            split_results[j, 6] = sum_g
            split_results[j, 7] = sum_h
            if sum_h < min_child_weight: return
            best_gain, best_bin = -1.0, -1
            left_g, left_h = 0.0, 0.0
            for b in range(max_bins - 1):
                left_g += histograms[j, b, 0]
                left_h += histograms[j, b, 1]
                right_g = sum_g - left_g
                right_h = sum_h - left_h
                if left_h < min_child_weight or right_h < min_child_weight: continue
                gain = (left_g**2 / (left_h + reg_lambda) +
                        right_g**2 / (right_h + reg_lambda) -
                        sum_g**2 / (sum_h + reg_lambda))
                if gain > best_gain:
                    best_gain, best_bin = gain, b
            if best_gain > gamma:
                split_results[j, 0], split_results[j, 1] = best_gain, best_bin
                split_results[j, 2], split_results[j, 3] = left_g, left_h
                split_results[j, 4], split_results[j, 5] = right_g, right_h

    @cuda.jit
    def _partition_nodes_cuda(X_bin, node_indices, best_feature, best_bin, target_node_id, left_node_id, right_node_id, n_samples):
        i = cuda.grid(1)
        if i < n_samples:
            if node_indices[i] == target_node_id:
                if X_bin[i, best_feature] <= best_bin: node_indices[i] = left_node_id
                else: node_indices[i] = right_node_id

    @cuda.jit
    def _update_F_cuda(X_bin, features, thresholds, values, left, right, learning_rate, F, n_samples):
        i = cuda.grid(1)
        if i < n_samples:
            node = 0
            while features[node] >= 0:
                if X_bin[i, features[node]] <= thresholds[node]: node = left[node]
                else: node = right[node]
            F[i] += learning_rate * values[node]

    @cuda.jit
    def _predict_all_trees_cuda(X_bin, features, thresholds, values, left, right, offsets, n_trees, learning_rate, out_preds, n_samples):
        i = cuda.grid(1)
        if i < n_samples:
            pred = 0.0
            for t in range(n_trees):
                node = offsets[t]
                while features[node] >= 0:
                    if X_bin[i, features[node]] <= thresholds[node]: node = left[node]
                    else: node = right[node]
                pred += learning_rate * values[node]
            out_preds[i] = pred

    # ═══════════════════════════════════════════════════════════════════════
    # BSSCL-GBM GPU — FULL FEATURE PARITY (PHASE 1)
    # Features: Binary, Multiclass, Regression, Verbose, Feature Importance,
    #           Early Stopping, Monotonic Constraints, SHAP
    # ═══════════════════════════════════════════════════════════════════════
    class BSSCL_GBM_GPU(HybridHistGBMNumbaV2):
        """
        Industrial-grade CUDA GPU execution engine for BSSCL-GBM.
        Supports Binary, Multiclass (Softmax), and Regression objectives.
        Includes verbose output, feature importance, early stopping,
        monotonic constraints, and SHAP explainability.
        """

        def _bin_data_gpu(self, X):
            """Bin features into uint8 histogram bins on CPU."""
            n_samples, n_features = X.shape
            X_bin = np.zeros((n_samples, n_features), dtype=np.uint8)
            self.edges = []
            for j in range(n_features):
                col = X[:, j]
                valid = col[~np.isnan(col)]
                bins = np.unique(np.percentile(valid, np.linspace(0, 100, self.max_bins)))
                self.edges.append(bins)
                safe_col = np.where(np.isnan(col), np.nanmedian(col), col)
                X_bin[:, j] = np.clip(np.digitize(safe_col, bins) - 1, 0, len(bins) - 1)
            return X_bin

        def _bin_predict_gpu(self, X):
            """Bin features for prediction using stored edges."""
            n_samples = len(X)
            X_bin = np.zeros((n_samples, X.shape[1]), dtype=np.uint8)
            for j in range(X.shape[1]):
                col = X[:, j]
                safe_col = np.where(np.isnan(col), np.nanmedian(col), col)
                X_bin[:, j] = np.clip(np.digitize(safe_col, self.edges[j]) - 1, 0, len(self.edges[j]) - 1)
            return X_bin

        def _gpu_predict_raw(self, X_bin, trees):
            """Run GPU prediction on binned data, returning raw scores."""
            n_samples = len(X_bin)
            n_trees = len(trees)
            max_nodes = len(trees[0]['features'])
            f_f = np.zeros(n_trees * max_nodes, dtype=np.int32)
            f_t = np.zeros(n_trees * max_nodes, dtype=np.float32)
            f_v = np.zeros(n_trees * max_nodes, dtype=np.float32)
            f_l = np.zeros(n_trees * max_nodes, dtype=np.int32)
            f_r = np.zeros(n_trees * max_nodes, dtype=np.int32)
            offsets = np.arange(0, n_trees * max_nodes, max_nodes, dtype=np.int32)
            for t, tree in enumerate(trees):
                s, e = t * max_nodes, (t + 1) * max_nodes
                f_f[s:e], f_t[s:e], f_v[s:e] = tree['features'], tree['thresholds'], tree['values']
                vl, vr = tree['left'] >= 0, tree['right'] >= 0
                f_l[s:e][vl], f_l[s:e][~vl] = tree['left'][vl] + s, -1
                f_r[s:e][vr], f_r[s:e][~vr] = tree['right'][vr] + s, -1
            d_out = cuda.device_array(n_samples, dtype=np.float32)
            threads, blocks = 256, (n_samples + 255) // 256
            _predict_all_trees_cuda[blocks, threads](
                cuda.to_device(X_bin), cuda.to_device(f_f), cuda.to_device(f_t),
                cuda.to_device(f_v), cuda.to_device(f_l), cuda.to_device(f_r),
                cuda.to_device(offsets), n_trees, self.learning_rate, d_out, n_samples
            )
            return d_out.copy_to_host()

        def _enforce_monotonicity_gpu(self, tree):
            """Post-process tree to enforce monotonic constraints on leaf values."""
            mono = getattr(self, 'monotone_constraints', None)
            if mono is None:
                return
            max_nodes = len(tree['features'])
            for node in range(max_nodes):
                feat = tree['features'][node]
                if feat < 0:
                    continue  # leaf node
                if feat >= len(mono) or mono[feat] == 0:
                    continue  # unconstrained
                lc = tree['left'][node]
                rc = tree['right'][node]
                if lc < 0 or rc < 0:
                    continue
                lv = tree['values'][lc]
                rv = tree['values'][rc]
                if mono[feat] == 1:  # increasing: left <= right
                    if lv > rv:
                        avg = (lv + rv) / 2.0
                        tree['values'][lc] = avg
                        tree['values'][rc] = avg
                elif mono[feat] == -1:  # decreasing: left >= right
                    if lv < rv:
                        avg = (lv + rv) / 2.0
                        tree['values'][lc] = avg
                        tree['values'][rc] = avg

        def fit(self, X, y, eval_set=None, X_val=None, y_val=None, group=None, group_val=None, chunk_size=None):
            """
            Train BSSCL-GBM on GPU with full feature parity.
            
            Parameters
            ----------
            X : np.ndarray - Training features
            y : np.ndarray - Training labels
            eval_set : list of tuples, optional - [(X_val, y_val)] for early stopping
            X_val, y_val : np.ndarray, optional - Validation data (legacy API)
            group : array-like, optional - Query group sizes for LambdaRank
            group_val : array-like, optional - Validation query group sizes
            chunk_size : int, optional - Rows per chunk for Out-of-Core (100GB+) support
            """
            if not CUDA_AVAILABLE:
                raise RuntimeError("CUDA is not available. Enable an Nvidia GPU.")

            t_start = time.time()

            # ── Categorical Features ──
            if getattr(self, 'categorical_features', None):
                X = self._encode_categorical(X, is_fit=True)
                if eval_set is not None and len(eval_set) > 0:
                    eval_set = [(self._encode_categorical(eval_set[0][0], is_fit=False), eval_set[0][1])]
                if X_val is not None:
                    X_val = self._encode_categorical(X_val, is_fit=False)

            # ── Handle eval_set ──
            if eval_set is not None and len(eval_set) > 0:
                X_val, y_val = eval_set[0]

            # ── Detect objective ──
            obj = getattr(self, 'objective', 'auto')
            if obj == 'regression':
                self._is_multiclass = False
                self._is_regression = True
                self._is_lambdarank = False
                self.n_classes_ = 1
                self.classes_ = None
                y = np.array(y, dtype=np.float64)
                if y_val is not None:
                    y_val = np.array(y_val, dtype=np.float64)
            elif obj == 'lambdarank':
                self._is_multiclass = False
                self._is_regression = False
                self._is_lambdarank = True
                self.n_classes_ = 1
                self.classes_ = None
                y = np.array(y, dtype=np.float64)
                if y_val is not None:
                    y_val = np.array(y_val, dtype=np.float64)
            else:
                self._is_regression = False
                self._is_lambdarank = False
                self.classes_ = np.unique(y)
                self.n_classes_ = len(self.classes_)
                self._is_multiclass = self.n_classes_ > 2
                label_map = {c: i for i, c in enumerate(self.classes_)}
                y = np.array([label_map[v] for v in y], dtype=np.int64)
                if y_val is not None:
                    y_val = np.array([label_map[v] for v in y_val], dtype=np.int64)

            # ── Resolve hyperparameters ──
            lr = getattr(self, 'learning_rate', 'auto')
            self.learning_rate = 0.08 if lr == 'auto' else lr
            md = getattr(self, 'max_depth', 'auto')
            self.max_depth = 6 if md == 'auto' else md
            ne = getattr(self, 'n_estimators', 'auto')
            self.n_estimators = 100 if ne == 'auto' else ne
            self.reg_lambda = getattr(self, 'reg_lambda', 1.0)
            gm = getattr(self, 'gamma', 'auto')
            self.gamma = 0.0 if gm == 'auto' else gm
            mcw = getattr(self, 'min_child_weight', 'auto')
            self.min_child_weight = 3.0 if mcw == 'auto' else mcw
            ss = getattr(self, 'subsample', 'auto')
            self.subsample = 0.75 if ss == 'auto' else ss
            cs = getattr(self, 'colsample_bytree', 'auto')
            self.colsample_bytree = 0.8 if cs == 'auto' else cs
            verbose = getattr(self, 'verbose', True)
            esr = getattr(self, 'early_stopping_rounds', None)
            loss_type = getattr(self, 'loss', 'mse')

            self.max_bins = 256
            self.trees = []
            self.train_loss_history_ = []
            self.val_loss_history_ = []

            n_samples, n_features = X.shape
            max_nodes = (1 << (self.max_depth + 1)) - 1

            # ── Mode string ──
            if self._is_regression:
                mode_str = "Regression (GPU)"
            elif self._is_lambdarank:
                mode_str = "LambdaRank (GPU)"
            elif self._is_multiclass:
                mode_str = f"Multi-class ({self.n_classes_} classes, GPU)"
                self.mc_trees = [[] for _ in range(self.n_estimators)]
            else:
                mode_str = "Binary (GPU)"

            # ── Out of Core Setup ──
            self._out_of_core = False
            if chunk_size and n_samples > chunk_size:
                self._out_of_core = True
                self._chunk_size = chunk_size
                mode_str += " [OUT-OF-CORE]"

            # ── Verbose: Print banner ──
            if verbose:
                print("\n" + "=" * 70)
                print(f"  Training BSSCL-GBM GPU — {mode_str}")
                print("=" * 70)
                print("  🔧 Hyperparameters:")
                print(f"     n_estimators:  {self.n_estimators}")
                print(f"     max_depth:     {self.max_depth}")
                print(f"     learning_rate: {self.learning_rate:.4f}")
                print(f"     n_bins:        {self.max_bins}")
                print(f"     gamma:         {self.gamma:.2f}")
                print(f"     min_child_w:   {self.min_child_weight}")
                print(f"     subsample:     {self.subsample}")
                print(f"     colsample:     {self.colsample_bytree}")
                if esr:
                    print(f"     early_stop:    {esr} rounds")
                if getattr(self, 'categorical_features', None):
                    print(f"     categorical:   {len(self.categorical_features)} features encoded")
                if self._out_of_core:
                    print(f"     OOC chunking:  {self._chunk_size} rows per block")

            # ── Binning ──
            t_bin = time.time()
            if self._out_of_core:
                # Subsample to build edges quickly
                sample_idx = np.random.choice(n_samples, min(100000, n_samples), replace=False)
                _ = self._bin_data_gpu(X[sample_idx]) # just to set self.edges
                
                # Create memmap for full binned data
                self._temp_dir = tempfile.mkdtemp()
                self._X_bin_path = os.path.join(self._temp_dir, 'X_bin.dat')
                X_bin = np.memmap(self._X_bin_path, dtype=np.uint8, mode='w+', shape=(n_samples, n_features))
                
                # Bin in chunks
                for start_idx in range(0, n_samples, self._chunk_size):
                    end_idx = min(start_idx + self._chunk_size, n_samples)
                    X_bin[start_idx:end_idx] = self._bin_predict_gpu(X[start_idx:end_idx])
                X_bin.flush()
            else:
                X_bin = self._bin_data_gpu(X)
                
            binning_ms = (time.time() - t_bin) * 1000

            if X_val is not None:
                X_val_bin = self._bin_predict_gpu(X_val)

            # ── Feature Importance Init ──
            self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
            self.feature_split_counts_ = np.zeros(n_features, dtype=np.int64)

            # ── Class Weights ──
            cw = getattr(self, 'class_weight', None)
            if self._is_regression or self._is_lambdarank:
                pass  # no class weights
            elif self._is_multiclass:
                w_c = np.ones(self.n_classes_, dtype=np.float32)
                if cw == 'balanced_sqrt':
                    for c in range(self.n_classes_):
                        n_c = np.sum(y == c)
                        w_c[c] = np.sqrt(n_samples / (self.n_classes_ * n_c)) if n_c > 0 else 1.0
                    if verbose:
                        weights_str = ", ".join([f"class_{c}={w_c[c]:.3f}" for c in range(self.n_classes_)])
                        print(f"  ⚖️  Class weights (sqrt-dampened): {weights_str}")
            else:
                mask_1 = y == 1
                n_1 = np.sum(mask_1)
                n_0 = n_samples - n_1
                w_1 = np.sqrt(n_0 / n_1) if (n_1 > 0 and cw == 'balanced_sqrt') else 1.0
                if cw == 'balanced_sqrt' and verbose:
                    w_0 = np.sqrt(n_1 / n_0) if n_0 > 0 else 1.0
                    print(f"  ⚖️  Class weights (sqrt-dampened): class_0={w_0:.3f}, class_1={w_1:.3f}")

            # ── Initialize F ──
            if self._is_regression:
                F_init = np.mean(y).astype(np.float32)
                self._F_init = F_init
                if self._out_of_core:
                    self._F_path = os.path.join(self._temp_dir, 'F.dat')
                    F = np.memmap(self._F_path, dtype=np.float32, mode='w+', shape=(n_samples,))
                    F[:] = F_init
                    d_F = None
                else:
                    F = np.full(n_samples, F_init, dtype=np.float32)
                    d_F = cuda.to_device(F)
            elif self._is_lambdarank:
                if self._out_of_core:
                    self._F_path = os.path.join(self._temp_dir, 'F.dat')
                    F = np.memmap(self._F_path, dtype=np.float32, mode='w+', shape=(n_samples,))
                    F[:] = 0.0
                    d_F = None
                else:
                    F = np.zeros(n_samples, dtype=np.float32)
                    d_F = cuda.to_device(F)
            elif self._is_multiclass:
                if self._out_of_core:
                    self._F_path = os.path.join(self._temp_dir, 'F.dat')
                    F = np.memmap(self._F_path, dtype=np.float32, mode='w+', shape=(n_samples, self.n_classes_))
                    F[:] = 0.0
                    d_F = None
                else:
                    F = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
                    d_F = [cuda.to_device(F[:, c].copy()) for c in range(self.n_classes_)]
            else:
                if self._out_of_core:
                    self._F_path = os.path.join(self._temp_dir, 'F.dat')
                    F = np.memmap(self._F_path, dtype=np.float32, mode='w+', shape=(n_samples,))
                    F[:] = 0.0
                    d_F = None
                else:
                    F = np.zeros(n_samples, dtype=np.float32)
                    d_F = cuda.to_device(F)

            if not self._out_of_core:
                d_X_bin = cuda.to_device(X_bin)
            d_node_indices = cuda.device_array(n_samples, dtype=np.int32)
            d_hist = cuda.device_array((n_features, self.max_bins, 2), dtype=np.float32)
            d_split_results = cuda.device_array((n_features, 8), dtype=np.float32)

            rng = np.random.RandomState(getattr(self, 'random_state', 42))
            threads = 256

            # ── Early Stopping State ──
            best_val_loss = np.inf
            best_iteration_ = 0
            rounds_no_improve = 0

            t_train = time.time()

            # ══════════════════════════════════════════════════════════════
            # BOOSTING ROUNDS
            # ══════════════════════════════════════════════════════════════
            actual_rounds = 0
            for round_idx in range(self.n_estimators):
                actual_rounds = round_idx + 1

                # ── Compute Gradients ──
                if self._is_regression:
                    F_host = np.array(F) if self._out_of_core else d_F.copy_to_host()
                    if loss_type == 'mae':
                        grad = np.sign(F_host - y).astype(np.float32)
                        hess = np.ones(n_samples, dtype=np.float32)
                    else:  # mse
                        grad = (F_host - y).astype(np.float32)
                        hess = np.ones(n_samples, dtype=np.float32)
                    train_loss = float(np.mean((F_host - y) ** 2))
                elif self._is_lambdarank:
                    F_host = np.array(F) if self._out_of_core else d_F.copy_to_host()
                    grad, hess = _compute_lambdarank_gradients(y, F_host, group)
                    grad = grad.astype(np.float32)
                    hess = hess.astype(np.float32)
                    train_loss = float(self._compute_ndcg(y, F_host, group))
                elif self._is_multiclass:
                    if self._out_of_core:
                        F_host = np.array(F)
                    else:
                        F_host = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
                        for c in range(self.n_classes_):
                            F_host[:, c] = d_F[c].copy_to_host()
                    exp_F = np.exp(F_host - np.max(F_host, axis=1, keepdims=True))
                    P = exp_F / np.sum(exp_F, axis=1, keepdims=True)
                    train_loss = float(-np.mean(np.log(P[np.arange(n_samples), y.astype(int)] + 1e-15)))
                else:
                    F_host = np.array(F) if self._out_of_core else d_F.copy_to_host()
                    p = 1.0 / (1.0 + np.exp(-F_host))
                    grad = (p - y).astype(np.float32)
                    hess = (p * (1.0 - p)).astype(np.float32)
                    if w_1 != 1.0:
                        grad[mask_1] *= w_1
                        hess[mask_1] *= w_1
                    train_loss = float(-np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)))

                self.train_loss_history_.append(train_loss)

                # ── Verbose: Progress ──
                if verbose and (round_idx == 0 or (round_idx + 1) % 20 == 0 or round_idx == self.n_estimators - 1):
                    if self._is_multiclass:
                        n_trees_per_round = self.n_classes_
                        print(f"  ✓ Round {round_idx+1:>3}/{self.n_estimators}  |  "
                              f"{n_trees_per_round} trees  |  train-loss: {train_loss:.6f}")
                    else:
                        print(f"  ✓ Tree  {round_idx+1:>3}/{self.n_estimators}  |  train-loss: {train_loss:.6f}")

                # ── Row & Feature Bagging ──
                n_sub_samples = max(1, int(n_samples * self.subsample))
                row_subset = rng.choice(n_samples, n_sub_samples, replace=False).astype(np.int32)
                n_sub_features = max(1, int(n_features * self.colsample_bytree))
                feature_subset = rng.choice(n_features, n_sub_features, replace=False).astype(np.int32)

                d_row_subset = cuda.to_device(row_subset)
                d_feature_subset = cuda.to_device(feature_subset)
                blocks_sub = (n_sub_samples + threads - 1) // threads
                blocks_all = (n_samples + threads - 1) // threads
                blocks_feat = (n_sub_features + threads - 1) // threads

                class_iters = range(self.n_classes_) if self._is_multiclass else [0]

                for c in class_iters:
                    if self._is_multiclass:
                        grad = (P[:, c] - (y == c)).astype(np.float32)
                        hess = (P[:, c] * (1.0 - P[:, c])).astype(np.float32)
                        if cw == 'balanced_sqrt' and w_c[c] != 1.0:
                            grad *= w_c[c]
                            hess *= w_c[c]
                        d_grad = cuda.to_device(grad)
                        d_hess = cuda.to_device(hess)
                    elif not self._is_regression:
                        d_grad = cuda.to_device(grad)
                        d_hess = cuda.to_device(hess)
                    else:
                        d_grad = cuda.to_device(grad)
                        d_hess = cuda.to_device(hess)

                    d_node_indices.copy_to_device(np.zeros(n_samples, dtype=np.int32))

                    tree = {'features': np.full(max_nodes, -1, dtype=np.int32),
                            'thresholds': np.zeros(max_nodes, dtype=np.float32),
                            'values': np.zeros(max_nodes, dtype=np.float32),
                            'left': np.full(max_nodes, -1, dtype=np.int32),
                            'right': np.full(max_nodes, -1, dtype=np.int32)}

                    active_nodes = [(0, 0)]

                    while active_nodes:
                        node, depth = active_nodes.pop(0)

                        d_hist.copy_to_device(np.zeros((n_features, self.max_bins, 2), dtype=np.float32))
                        
                        if self._out_of_core:
                            # Build histograms in chunks directly from memmap
                            for start_idx in range(0, n_samples, self._chunk_size):
                                end_idx = min(start_idx + self._chunk_size, n_samples)
                                chunk_size_n = end_idx - start_idx
                                
                                # Filter row_subset to this chunk
                                chunk_mask = (row_subset >= start_idx) & (row_subset < end_idx)
                                chunk_rows = row_subset[chunk_mask] - start_idx
                                n_chunk_samples = len(chunk_rows)
                                
                                if n_chunk_samples == 0:
                                    continue
                                    
                                d_X_bin_chunk = cuda.to_device(X_bin[start_idx:end_idx])
                                d_grad_chunk = cuda.to_device(grad[start_idx:end_idx])
                                d_hess_chunk = cuda.to_device(hess[start_idx:end_idx])
                                d_node_indices_chunk = cuda.to_device(d_node_indices.copy_to_host()[start_idx:end_idx])
                                d_row_subset_chunk = cuda.to_device(chunk_rows)
                                
                                blocks_chunk = (n_chunk_samples + threads - 1) // threads
                                
                                _build_histograms_single_node_cuda[blocks_chunk, threads](
                                    d_X_bin_chunk, d_grad_chunk, d_hess_chunk, d_hist, d_node_indices_chunk,
                                    d_row_subset_chunk, d_feature_subset,
                                    n_chunk_samples, n_sub_features, node
                                )
                        else:
                            _build_histograms_single_node_cuda[blocks_sub, threads](
                                d_X_bin, d_grad, d_hess, d_hist, d_node_indices,
                                d_row_subset, d_feature_subset,
                                n_sub_samples, n_sub_features, node
                            )

                        d_split_results.copy_to_device(np.zeros((n_features, 8), dtype=np.float32))
                        _find_best_split_from_hist_cuda[blocks_feat, threads](
                            d_hist, d_split_results, d_feature_subset,
                            n_sub_features, self.max_bins,
                            self.reg_lambda, self.gamma, self.min_child_weight
                        )

                        res = d_split_results.copy_to_host()
                        best_feat, best_gain = -1, -1.0
                        for f_idx in range(n_sub_features):
                            j = feature_subset[f_idx]
                            if res[j, 0] > best_gain:
                                best_gain, best_feat = res[j, 0], j

                        if best_feat != -1:
                            sum_g, sum_h = res[best_feat, 6], res[best_feat, 7]
                        else:
                            sum_g, sum_h = 0.0, 0.0
                            for f_idx in range(n_sub_features):
                                j = feature_subset[f_idx]
                                if res[j, 7] > 0:
                                    sum_g, sum_h = res[j, 6], res[j, 7]
                                    break

                        multiplier = (self.n_classes_ - 1) / self.n_classes_ if self._is_multiclass else 1.0
                        tree['values'][node] = -sum_g / (sum_h * multiplier + self.reg_lambda) if sum_h > 0 else 0.0

                        can_split = (best_feat != -1 and (node * 2 + 2) < max_nodes and depth < self.max_depth)

                        if can_split:
                            best_bin = int(res[best_feat, 1])
                            lc, rc = node * 2 + 1, node * 2 + 2
                            tree['features'][node] = best_feat
                            tree['thresholds'][node] = best_bin
                            tree['left'][node] = lc
                            tree['right'][node] = rc
                            tree['values'][lc] = -res[best_feat, 2] / (res[best_feat, 3] * multiplier + self.reg_lambda)
                            tree['values'][rc] = -res[best_feat, 4] / (res[best_feat, 5] * multiplier + self.reg_lambda)
                            
                            if self._out_of_core:
                                node_idx_host = d_node_indices.copy_to_host()
                                for start_idx in range(0, n_samples, self._chunk_size):
                                    end_idx = min(start_idx + self._chunk_size, n_samples)
                                    d_X_bin_chunk = cuda.to_device(X_bin[start_idx:end_idx])
                                    d_node_indices_chunk = cuda.to_device(node_idx_host[start_idx:end_idx])
                                    chunk_size_n = end_idx - start_idx
                                    blocks_chunk = (chunk_size_n + threads - 1) // threads
                                    _partition_nodes_cuda[blocks_chunk, threads](
                                        d_X_bin_chunk, d_node_indices_chunk, best_feat, best_bin, node, lc, rc, chunk_size_n
                                    )
                                    node_idx_host[start_idx:end_idx] = d_node_indices_chunk.copy_to_host()
                                d_node_indices.copy_to_device(node_idx_host)
                            else:
                                _partition_nodes_cuda[blocks_all, threads](
                                    d_X_bin, d_node_indices, best_feat, best_bin, node, lc, rc, n_samples
                                )
                                
                            active_nodes.extend([(lc, depth + 1), (rc, depth + 1)])

                            # ── Feature Importance: Accumulate ──
                            self.feature_importances_[best_feat] += best_gain
                            self.feature_split_counts_[best_feat] += 1
                        else:
                            tree['features'][node] = -1

                    # ── Monotonic Constraints: Post-process ──
                    self._enforce_monotonicity_gpu(tree)

                    if self._is_multiclass:
                        self.mc_trees[round_idx].append(tree)
                        if self._out_of_core:
                            for start_idx in range(0, n_samples, self._chunk_size):
                                end_idx = min(start_idx + self._chunk_size, n_samples)
                                d_X_bin_chunk = cuda.to_device(X_bin[start_idx:end_idx])
                                d_F_chunk = cuda.to_device(np.array(F[start_idx:end_idx, c]))
                                chunk_size_n = end_idx - start_idx
                                blocks_chunk = (chunk_size_n + threads - 1) // threads
                                _update_F_cuda[blocks_chunk, threads](
                                    d_X_bin_chunk, cuda.to_device(tree['features']), cuda.to_device(tree['thresholds']),
                                    cuda.to_device(tree['values']), cuda.to_device(tree['left']), cuda.to_device(tree['right']),
                                    self.learning_rate, d_F_chunk, chunk_size_n
                                )
                                F[start_idx:end_idx, c] = d_F_chunk.copy_to_host()
                        else:
                            _update_F_cuda[blocks_all, threads](
                                d_X_bin, cuda.to_device(tree['features']), cuda.to_device(tree['thresholds']),
                                cuda.to_device(tree['values']), cuda.to_device(tree['left']), cuda.to_device(tree['right']),
                                self.learning_rate, d_F[c], n_samples
                            )
                    else:
                        self.trees.append(tree)
                        if self._out_of_core:
                            for start_idx in range(0, n_samples, self._chunk_size):
                                end_idx = min(start_idx + self._chunk_size, n_samples)
                                d_X_bin_chunk = cuda.to_device(X_bin[start_idx:end_idx])
                                d_F_chunk = cuda.to_device(np.array(F[start_idx:end_idx]))
                                chunk_size_n = end_idx - start_idx
                                blocks_chunk = (chunk_size_n + threads - 1) // threads
                                _update_F_cuda[blocks_chunk, threads](
                                    d_X_bin_chunk, cuda.to_device(tree['features']), cuda.to_device(tree['thresholds']),
                                    cuda.to_device(tree['values']), cuda.to_device(tree['left']), cuda.to_device(tree['right']),
                                    self.learning_rate, d_F_chunk, chunk_size_n
                                )
                                F[start_idx:end_idx] = d_F_chunk.copy_to_host()
                            if self._is_lambdarank or self._is_regression:
                                F.flush()
                        else:
                            _update_F_cuda[blocks_all, threads](
                                d_X_bin, cuda.to_device(tree['features']), cuda.to_device(tree['thresholds']),
                                cuda.to_device(tree['values']), cuda.to_device(tree['left']), cuda.to_device(tree['right']),
                                self.learning_rate, d_F, n_samples
                            )

                # ── Early Stopping: Validation Check ──
                if esr and X_val is not None and y_val is not None:
                    if self._is_regression:
                        val_raw = self._gpu_predict_raw(X_val_bin, self.trees)
                        val_raw += getattr(self, '_F_init', 0.0)
                        val_loss = float(np.mean((val_raw - y_val) ** 2))
                    elif self._is_lambdarank:
                        val_raw = self._gpu_predict_raw(X_val_bin, self.trees)
                        val_loss = -float(self._compute_ndcg(y_val, val_raw, group_val)) # negative for minimization logic
                    elif self._is_multiclass:
                        val_F = np.zeros((len(X_val), self.n_classes_), dtype=np.float32)
                        for vc in range(self.n_classes_):
                            c_trees = [r[vc] for r in self.mc_trees[:round_idx + 1]]
                            val_F[:, vc] = self._gpu_predict_raw(X_val_bin, c_trees)
                        exp_vF = np.exp(val_F - np.max(val_F, axis=1, keepdims=True))
                        val_P = exp_vF / np.sum(exp_vF, axis=1, keepdims=True)
                        val_loss = float(-np.mean(np.log(val_P[np.arange(len(y_val)), y_val.astype(int)] + 1e-15)))
                    else:
                        val_raw = self._gpu_predict_raw(X_val_bin, self.trees)
                        val_p = 1.0 / (1.0 + np.exp(-val_raw))
                        val_loss = float(-np.mean(y_val * np.log(val_p + 1e-15) + (1 - y_val) * np.log(1 - val_p + 1e-15)))

                    self.val_loss_history_.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_iteration_ = round_idx + 1
                        rounds_no_improve = 0
                    else:
                        rounds_no_improve += 1

                    if rounds_no_improve >= esr:
                        if verbose:
                            print(f"  ⏹ Early stopping at round {round_idx + 1} (best: {best_iteration_})")
                        # Trim trees
                        if self._is_multiclass:
                            self.mc_trees = self.mc_trees[:best_iteration_]
                        else:
                            self.trees = self.trees[:best_iteration_]
                        break

            # ── Normalize Feature Importance ──
            total_gain = np.sum(self.feature_importances_)
            if total_gain > 0:
                self.feature_importances_ = self.feature_importances_ / total_gain

            training_ms = (time.time() - t_train) * 1000
            total_ms = (time.time() - t_start) * 1000

            # ── Verbose: Summary ──
            if verbose:
                total_trees = actual_rounds * (self.n_classes_ if self._is_multiclass else 1)
                print(f"\n  ✓ Training complete! ({total_trees} trees)")
                print(f"    Binning:  {binning_ms:.2f} ms")
                print(f"    Training: {training_ms:.2f} ms")
                print(f"    Total:    {total_ms:.2f} ms")

            self.n_features_in_ = n_features
            self.best_iteration_ = best_iteration_ if esr else actual_rounds
            return self

        def predict(self, X):
            """Predict class labels (classification) or values (regression)."""
            if getattr(self, 'categorical_features', None):
                X = self._encode_categorical(X, is_fit=False)
            if self._is_regression:
                X_bin = self._bin_predict_gpu(X)
                raw = self._gpu_predict_raw(X_bin, self.trees)
                return raw + getattr(self, '_F_init', 0.0)
            elif self._is_lambdarank:
                X_bin = self._bin_predict_gpu(X)
                return self._gpu_predict_raw(X_bin, self.trees)
            else:
                proba = self.predict_proba(X)
                return self.classes_[np.argmax(proba, axis=1)]

        def predict_proba(self, X):
            """Predict class probabilities."""
            if getattr(self, 'categorical_features', None):
                X = self._encode_categorical(X, is_fit=False)
            X_bin = self._bin_predict_gpu(X)
            n_samples = len(X)
            d_X_bin = cuda.to_device(X_bin)
            threads, blocks = 256, (n_samples + 255) // 256

            if self._is_multiclass:
                F = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
                for c in range(self.n_classes_):
                    class_trees = [r[c] for r in self.mc_trees]
                    F[:, c] = self._gpu_predict_raw(X_bin, class_trees)
                exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
                return exp_F / np.sum(exp_F, axis=1, keepdims=True)
            else:
                raw = self._gpu_predict_raw(X_bin, self.trees)
                p = 1.0 / (1.0 + np.exp(-raw))
                return np.vstack([1.0 - p, p]).T

        def plot_feature_importance(self, top_n=20, figsize=(10, 6)):
            """Plot feature importance (gain-based)."""
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not available for plotting.")
                return

            n_features = len(self.feature_importances_)
            names = [f"Feature {i}" for i in range(n_features)]
            indices = np.argsort(self.feature_importances_)[::-1][:top_n]

            plt.figure(figsize=figsize)
            plt.title("BSSCL-GBM GPU — Feature Importance (Gain)")
            plt.bar(range(len(indices)), self.feature_importances_[indices])
            plt.xticks(range(len(indices)), [names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        def get_shap_explainer(self):
            """Return a SHAP TreeExplainer for this model."""
            try:
                import shap
            except ImportError:
                raise ImportError("Please install shap: pip install shap")

            if self._is_multiclass:
                all_trees = []
                for round_trees in self.mc_trees:
                    for tree in round_trees:
                        all_trees.append(tree)
            else:
                all_trees = self.trees

            n_nodes_list = [len(t['features']) for t in all_trees]
            max_n = max(n_nodes_list)

            children_left = np.full((len(all_trees), max_n), -1, dtype=np.int64)
            children_right = np.full((len(all_trees), max_n), -1, dtype=np.int64)
            features_arr = np.full((len(all_trees), max_n), -2, dtype=np.int64)
            thresholds_arr = np.full((len(all_trees), max_n), -2.0, dtype=np.float64)
            values_arr = np.zeros((len(all_trees), max_n, 1), dtype=np.float64)

            for t_idx, tree in enumerate(all_trees):
                n = len(tree['features'])
                for node in range(n):
                    if tree['features'][node] >= 0:
                        features_arr[t_idx, node] = tree['features'][node]
                        lc, rc = tree['left'][node], tree['right'][node]
                        children_left[t_idx, node] = lc
                        children_right[t_idx, node] = rc
                        if len(self.edges[tree['features'][node]]) > int(tree['thresholds'][node]):
                            thresholds_arr[t_idx, node] = self.edges[tree['features'][node]][int(tree['thresholds'][node])]
                        else:
                            thresholds_arr[t_idx, node] = tree['thresholds'][node]
                    else:
                        features_arr[t_idx, node] = -2
                        children_left[t_idx, node] = -1
                        children_right[t_idx, node] = -1
                    values_arr[t_idx, node, 0] = tree['values'][node] * self.learning_rate

            model_dict = {
                'children_left': children_left,
                'children_right': children_right,
                'children_default': children_left.copy(),
                'features': features_arr,
                'thresholds': thresholds_arr,
                'values': values_arr,
                'node_sample_weight': np.ones_like(values_arr),
            }

            if self._is_regression:
                model_dict['objective'] = 'squared_error'
                model_dict['base_values'] = np.array([getattr(self, '_F_init', 0.0)])
            elif self._is_multiclass:
                model_dict['objective'] = 'cross_entropy'
                model_dict['output_type'] = 'log_loss'
                model_dict['base_values'] = np.zeros(self.n_classes_)
                model_dict['num_outputs'] = self.n_classes_
            else:
                model_dict['objective'] = 'binary_crossentropy'
                model_dict['base_values'] = np.array([0.0])

            model_dict['tree_output'] = 'raw_value'
            model_dict['input_dtype'] = np.float32

            ensemble = shap.models.Model(model_dict)
            return shap.TreeExplainer(model_dict)

else:
    class BSSCL_GBM_GPU(HybridHistGBMNumbaV2):
        def fit(self, X, y, **kwargs):
            raise RuntimeError("CUDA is not available. Please enable an Nvidia GPU.")
