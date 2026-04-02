#---------------------------
# IMPORTS & HYPERPARAMETERS
# ---------------------------
import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_recall_curve
# deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import networkx as nx
import random
from collections import deque

# optional
try:
    import umap
except Exception:
    umap = None

# Hyperparameters (edit as needed)
DATA_PATH = 'data.csv'  # expected to have node_id + features + optional 'attack'
RANDOM_SEED = 42
AE_EPOCHS = 50
AE_BATCH = 64
AE_LATENT = 8
IF_N_ESTIMATORS = 200
LOF_N_NEIGHBORS = 20
COMBINE_WEIGHTS = {'ae':0.4, 'if':0.3, 'lof':0.3}  # weighted ensemble for combined score

# Honeypot Simulation Parameters
WINDOW = 10
MAX_TIMESTEPS = 100
COST_SPAWN = 1.0
SPAWN_THRESHOLD = 0.3 
TOPK_SPAWN = 5


# ---------------------------
# UTILITIES
# ---------------------------
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED) # Set TF seed


def load_data(path=DATA_PATH):
    import pandas as pd 
    
    # Check for placeholder data if file doesn't exist (useful for testing the logic)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Generating placeholder data.")
        # Create a placeholder dataset with 12 core features + label
        n_samples = 500
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(n_samples)],
            'feat_1': np.random.rand(n_samples) * 10,
            'feat_2': np.random.randn(n_samples),
            'feat_3': np.random.randint(0, 5, n_samples),
            'feat_4': np.random.rand(n_samples) * 5,
            'feat_5': np.random.randn(n_samples),
            'feat_6': np.random.randint(0, 10, n_samples),
            'feat_7': np.random.rand(n_samples) * 2,
            'feat_8': np.random.randn(n_samples),
            'feat_9': np.random.randint(0, 2, n_samples),
            'feat_10': np.random.rand(n_samples) * 1,
            'feat_11': np.random.rand(n_samples) * 10,
            'feat_12': np.random.randn(n_samples),
            
            # Simulation features must be present in the original data for consistent scaling
            'exploit_attempts': np.zeros(n_samples), 
            'scan_count': np.zeros(n_samples),       
            
            'label': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        })
        # Introduce a clear anomaly pattern (e.g., high feat_1 for attacks)
        df.loc[df['label'] == 1, 'feat_1'] = df.loc[df['label'] == 1, 'feat_1'] * 3
    else:
        df = pd.read_csv(path)
        
    if 'node' not in df.columns:
        raise ValueError("CSV must contain a 'node' column")
    
    # CRITICAL FIX STEP 1: Add missing simulation columns to real data if not present
    if 'exploit_attempts' not in df.columns:
        df['exploit_attempts'] = 0.0
    if 'scan_count' not in df.columns:
        df['scan_count'] = 0.0
    
    features = df.drop(columns=['node'])
    labels = None
    if 'label' in features.columns:
        labels = features['label'].astype(int).values
        features = features.drop(columns=['label'])
    
    # Only keep numeric features for scaling
    features = features.select_dtypes(include=[np.number])
    
    return df['node'].values, features, labels


# ---------------------------
# PREPROCESSING
# ---------------------------

def preprocess(features: pd.DataFrame):
    # Basic cleaning: fillna
    features = features.copy()
    features = features.fillna(features.median())
    
    # We avoid dropping constant/low-variance columns here to maintain consistency 
    # between the fitted scaler and the dynamically updated features in the simulation.
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return X, scaler, features.columns.tolist()


# ---------------------------
# AUTOENCODER
# ---------------------------

def build_autoencoder(input_dim, latent_dim=AE_LATENT):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(32, input_dim//2), activation='relu')(inp)
    x = layers.Dense(max(16, input_dim//4), activation='relu')(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(max(16, input_dim//4), activation='relu')(latent)
    x = layers.Dense(max(32, input_dim//2), activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out, name='autoencoder')
    ae.compile(optimizer='adam', loss='mse')
    return ae


def train_autoencoder(X_train, X_val=None, epochs=AE_EPOCHS, batch_size=AE_BATCH):
    ae = build_autoencoder(X_train.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=8, restore_best_weights=True)
    if X_val is None:
        ae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    else:
        ae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    return ae


def ae_reconstruction_error(ae, X):
    recon = ae.predict(X, verbose=0)
    mse = np.mean(np.square(X - recon), axis=1)
    return mse


# ---------------------------
# ISOLATION FOREST
# ---------------------------

def train_if(X):
    clf = IsolationForest(n_estimators=IF_N_ESTIMATORS, random_state=RANDOM_SEED, contamination='auto', n_jobs=-1)
    clf.fit(X)
    scores = -clf.score_samples(X)  # higher -> more anomalous
    return clf, scores


# ---------------------------
# LOCAL OUTLIER FACTOR
# ---------------------------

def train_lof(X):
    # LOF doesn't have a separate predict-only fit; use fit_predict for scores
    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination='auto', novelty=True, n_jobs=-1)
    lof.fit(X)
    scores = -lof.decision_function(X)  # higher -> more anomalous
    return lof, scores


# ---------------------------
# ENSEMBLE & CLUSTERING
# ---------------------------

def combine_scores(ae_score, if_score, lof_score, weights=COMBINE_WEIGHTS):
    # normalize each score to 0-1
    def _norm(s):
        s = np.array(s)
        if s.max() - s.min() == 0:
            return np.zeros_like(s)
        return (s - s.min()) / (s.max() - s.min())
    na = _norm(ae_score)
    ni = _norm(if_score)
    nl = _norm(lof_score)
    combined = weights['ae']*na + weights['if']*ni + weights['lof']*nl
    return combined, na, ni, nl


def cluster_embeddings(X, method='dbscan'):
    if method == 'dbscan':
        c = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    else:
        c = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init='auto').fit_predict(X)
    return c


# ---------------------------
# EVALUATION (if labels exist)
# ---------------------------

def evaluate_scores(scores, labels):
    """
    Evaluate anomaly scores against true labels.
    Automatically finds threshold that maximizes F1.
    """
    if labels is None or len(np.unique(labels)) < 2:
        return {}

    scores = np.array(scores)

    # Use precision-recall curve to find best threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # avoid div0
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    preds = (scores >= best_thresh).astype(int)

    roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels))>1 else None
    precision_val = precision_score(labels, preds, zero_division=0)
    recall_val = recall_score(labels, preds, zero_division=0)
    f1_val = f1_score(labels, preds, zero_division=0)

    return {
        'roc_auc': roc_auc,
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1_val,
        'threshold': best_thresh
    }


# ---------------------------
# PLOTTING
# ---------------------------

def plot_roc(labels, scores_dict):
    if labels is None or len(np.unique(labels)) < 2:
        print("Skipping ROC plot: Labels not available or insufficient unique labels.")
        return
        
    plt.figure(figsize=(8,6))
    for name, s in scores_dict.items():
        fpr, tpr, _ = roc_curve(labels, s)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curves')
    plt.tight_layout()
    plt.show()


def plot_umap(X, scores, labels=None):
    if umap is None:
        print('UMAP not installed — skipping UMAP plot')
        return
    if X.shape[0] < 50:
        print('Not enough samples for UMAP plot — skipping.')
        return
        
    reducer = umap.UMAP(random_state=RANDOM_SEED)
    emb = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=scores, cmap='viridis', s=8)
    plt.colorbar(sc, label='anomaly score')
    if labels is not None and len(np.unique(labels)) > 1:
        # overlay attack points
        atk_idx = np.where(labels==1)[0]
        if len(atk_idx) > 0:
            plt.scatter(emb[atk_idx,0], emb[atk_idx,1], facecolors='none', edgecolors='r', s=50, label='attack')
            plt.legend()
    plt.title('UMAP embedding colored by anomaly score (Test Set)')
    plt.show()


# ---------------------------
# LOCAL TOPOLOGY HELPERS
# ---------------------------
def _local_subgraph(G, center_idx, max_hops=2):
    """Get nodes within max_hops distance from center_idx."""
    visited = set()
    queue = deque([(center_idx, 0)])
    visited.add(center_idx)
    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist + 1))
    return visited


def update_topology_on_honeypot_hit(
    G, hit_node_idx, X_unscaled_current, node_id_to_graph_map, graph_to_node_id_map, node_ids,
    rewiring_prob=0.6, add_new_nodes=True, max_new_nodes=2, max_hops=2, local_rewire_fraction=0.5
):
    """
    Localised topology adaptation + SYNCHRONIZE new nodes with feature DataFrame.
    """
    import random
    n_before = len(G.nodes)

    # 1. LOCAL REWIRING (only inside neighbourhood)
    if random.random() < rewiring_prob:
        local_nodes = _local_subgraph(G, hit_node_idx, max_hops=max_hops)
        local_nodes.discard(hit_node_idx)
        current_local_edges = [n for n in G.neighbors(hit_node_idx) if n in local_nodes]
        
        if current_local_edges:
            n_rewire = max(1, int(len(current_local_edges) * local_rewire_fraction))
            to_remove = random.sample(current_local_edges, min(n_rewire, len(current_local_edges)))
            
            for n in to_remove:
                G.remove_edge(hit_node_idx, n)
            
            possible_targets = list(local_nodes - set(to_remove))
            if possible_targets:
                n_add = len(to_remove)
                new_links = random.sample(possible_targets, min(n_add, len(possible_targets)))
                for t in new_links:
                    G.add_edge(hit_node_idx, t)
                
                print(f"[LOCAL TOPO] Node {hit_node_idx}: -{len(to_remove)} +{len(new_links)} edges (hops={max_hops})")

    # 2. ADD NEW HONEYPOT NODES + SYNC WITH DATAFRAME
    if add_new_nodes:
        num_new = random.randint(1, max_new_nodes)
        for _ in range(num_new):
            new_idx = max(G.nodes()) + 1
            new_id = f'new_hp_{new_idx}'
            
            # Add to graph
            G.add_node(new_idx, is_honeypot=True, node_id=new_id, value=0)
            connect_to = [hit_node_idx]
            neighbours = list(G.neighbors(hit_node_idx))
            if neighbours:
                extra = random.sample(neighbours, min(2, len(neighbours)))
                connect_to.extend(extra)
            for c in connect_to:
                G.add_edge(new_idx, c)
            
            # === CRITICAL: SYNC WITH FEATURES ===
            if new_id not in X_unscaled_current.index:
                default_row = pd.Series({col: 0.0 for col in X_unscaled_current.columns}, name=new_id)
                X_unscaled_current = pd.concat([X_unscaled_current, default_row.to_frame().T])
            
            # Update mappings
            node_id_to_graph_map[new_id] = new_idx
            graph_to_node_id_map[new_idx] = new_id
            if new_id not in node_ids:
                node_ids.append(new_id)
        
        print(f"[LOCAL TOPO] Added {num_new} honeypots near {hit_node_idx}. Nodes: {n_before}→{len(G.nodes())}")

    return X_unscaled_current, node_id_to_graph_map, graph_to_node_id_map, node_ids


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def run_all(save_outputs=True):
    import pandas as pd
    
    # Load original data (includes initial 0s for exploit_attempts/scan_count)
    node_ids, features_df, labels = load_data(DATA_PATH)
    
    # Store original features to enable splitting the unscaled data
    original_features_values = features_df.values
    original_feature_columns = features_df.columns.tolist()
    
    # Preprocess and scale the full dataset (scaler is fitted here)
    X, scaler, feat_names = preprocess(features_df)

    # ---------------------------
    # Train-test split 
    # ---------------------------
    split_inputs = [X, node_ids, original_features_values]
    if labels is not None and len(np.unique(labels)) > 1:
        split_inputs.insert(1, labels)
        splitter = train_test_split(
            *split_inputs, 
            test_size=0.3, random_state=RANDOM_SEED, stratify=labels
        )
        X_train, X_test, y_train, y_test, node_train, node_test, features_train_val, features_test_val = splitter
    else:
        splitter = train_test_split(
            *split_inputs, 
            test_size=0.3, random_state=RANDOM_SEED
        )
        X_train, X_test, node_train, node_test, features_train_val, features_test_val = splitter
        y_train = y_test = None

    # Create the DataFrame of unscaled test features for the simulation
    features_test_df = pd.DataFrame(features_test_val, columns=original_feature_columns)
    features_test_df.index = node_test # Set index to node ID

    # ---------------------------
    # AUTOENCODER
    # ---------------------------
    print('Training Autoencoder...')
    try:
        if y_train is not None and len(X_train[y_train == 0]) > 0:
            X_ae_train = X_train[y_train == 0]
        else:
            X_ae_train = X_train

        X_ae_val = X_ae_train[:max(1, int(0.1 * len(X_ae_train)))]
        
        ae = train_autoencoder(X_ae_train, X_val=X_ae_val)
        ae_scores = ae_reconstruction_error(ae, X_test)
    except Exception as e:
        print("Error training AE or computing scores:", e)
        ae_scores = np.zeros(X_test.shape[0])

    # ---------------------------
    # ISOLATION FOREST
    # ---------------------------
    print('Training Isolation Forest...')
    try:
        if_clf, _ = train_if(X_train) # Fit on training data
        if_scores = -if_clf.score_samples(X_test) # Score on test data
    except Exception as e:
        print("Error training IF or computing scores:", e)
        if_scores = np.zeros(X_test.shape[0])

    # ---------------------------
    # LOCAL OUTLIER FACTOR
    # ---------------------------
    print('Training LOF...')
    try:
        lof_clf, _ = train_lof(X_train) # Fit on training data
        lof_scores = -lof_clf.decision_function(X_test) # Score on test data
    except Exception as e:
        print("Error training LOF or computing scores:", e)
        lof_scores = np.zeros(X_test.shape[0])

    # ---------------------------
    # COMBINE SCORES (for evaluation/plotting only)
    # ---------------------------
    combined, na, ni, nl = combine_scores(ae_scores, if_scores, lof_scores)

    # ---------------------------
    # EVALUATION
    # ---------------------------
    eval_ae = evaluate_scores(ae_scores, y_test)
    eval_if = evaluate_scores(if_scores, y_test)
    eval_lof = evaluate_scores(lof_scores, y_test)
    eval_comb = evaluate_scores(combined, y_test)

    # ---------------------------
    # OUTPUT TABLE
    # ---------------------------
    out_df = pd.DataFrame({
        'node': node_test,
        'ae_score': ae_scores,
        'if_score': if_scores,
        'lof_score': lof_scores,
        'ae_norm': na,
        'if_norm': ni,
        'lof_norm': nl,
        'combined_score': combined
    })
    if y_test is not None:
        out_df['label'] = y_test

    if save_outputs:
        out_df.to_csv('anomaly_scores.csv', index=False)
        with open('evaluation_metrics.json', 'w') as f:
            json.dump({'ae': eval_ae, 'if': eval_if, 'lof': eval_lof, 'combined': eval_comb}, f, default=lambda o: None)

    # ---------------------------
    # PLOTS
    # ---------------------------
    plot_roc(y_test, {'Autoencoder': ae_scores, 'IF': if_scores, 'LOF': lof_scores, 'Combined': combined})
    plot_umap(X_test, combined, y_test)

    # ---------------------------
    # METRICS TABLE
    # ---------------------------
    metrics_df = pd.DataFrame({'AE': eval_ae, 'IF': eval_if, 'LOF': eval_lof, 'Combined': eval_comb}).T
    print("\nEvaluation Metrics (Test Set):")
    print(metrics_df[['roc_auc', 'precision', 'recall', 'f1']].round(3))

    # ---------------------------
    # CONFUSION MATRIX (AE)
    # ---------------------------
    if y_test is not None and len(eval_ae) > 0:
        threshold = eval_ae.get('threshold', 0.5)
        preds = (ae_scores >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (AE on Test Set)')
        plt.show()

    # Return necessary components for simulation + AE train MSE for threshold
    train_mse = ae_reconstruction_error(ae, X_train)
    ae_threshold = np.percentile(train_mse, 95)
    print(f"AE Spawn Threshold (95th percentile of train MSE): {ae_threshold:.4f}")
    
    return out_df, {'ae': eval_ae, 'if': eval_if, 'lof': eval_lof, 'combined': eval_comb}, ae, if_clf, lof_clf, features_test_df, scaler, ae_threshold


# ---------------------------
# SIMPLE HONEYPOT SIMULATION
# ---------------------------
class Controller:
    def __init__(self):
        self.cost = 0.0
    def apply(self, G, actions):
        for i in actions:
            if not G.nodes[i].get('is_honeypot', False):
                G.nodes[i]['is_honeypot'] = True
                self.cost += COST_SPAWN # use global hyperparameter

class RandomAttacker:
    def __init__(self, start_graph_node):
        self.pos = start_graph_node
    def step(self, G):
        neighbors = list(G.neighbors(self.pos))
        honeypot_neighbors = [n for n in neighbors if G.nodes[n].get('is_honeypot', False)]
        
        # Attacker movement logic 
        if honeypot_neighbors and random.random() < 0.5:
            self.pos = random.choice(honeypot_neighbors)
        elif neighbors and random.random() < 0.7:
            self.pos = random.choice(neighbors)
        else:
            self.pos = random.choice(list(G.nodes()))
            
        action = 'scan' if random.random() < 0.6 else 'exploit'
        return action, self.pos # self.pos is graph node index
    

def run_honeypot_sim_adaptive(features_test_df, ae_model, if_clf, lof_clf, scaler, feat_names, ae_threshold,
                              topk_spawn=8, max_timesteps=MAX_TIMESTEPS):
    """
    Adaptive honeypot simulation with LOCALISED topology changes + FULL SYNCHRONIZATION.
    Uses only AE scores for deployment decisions.
    """
    import pandas as pd
    
    # === INITIALIZE MUTABLE STATE ===
    X_unscaled_current = features_test_df.copy()
    node_ids = list(features_test_df.index)  # mutable list
    node_id_to_graph_map = {nid: i for i, nid in enumerate(node_ids)}
    graph_to_node_id_map = {i: nid for i, nid in enumerate(node_ids)}
    
    n_nodes = len(node_ids)
    
    # Create Graph
    G = nx.erdos_renyi_graph(n_nodes, 0.08, seed=RANDOM_SEED)

    # Initialize node attributes
    for i in G.nodes():
        node_id = graph_to_node_id_map[i]
        G.nodes[i]['node_id'] = node_id
        G.nodes[i]['is_honeypot'] = False
        G.nodes[i]['value'] = 1 if random.random() < 0.2 else 0

    # Initial honeypots
    INITIAL_HONEYPOTS = max(3, topk_spawn // 2)
    initial_hp_nodes = random.sample(list(G.nodes()), INITIAL_HONEYPOTS)
    for i in initial_hp_nodes:
        G.nodes[i]['is_honeypot'] = True

    controller = Controller()
    attackers = [RandomAttacker(0)] 
    compromised = set()
    stats = {'honeypot_hits': 0, 'real_hits': 0, 'cost': 0.0, 'timesteps': 0}

    print(f"\n=== Adaptive Simulation Started ===")
    print(f"Nodes: {n_nodes}, Initial honeypots: {INITIAL_HONEYPOTS}, Threshold: {ae_threshold:.4f}")
    
    for t in range(max_timesteps):
        stats['timesteps'] = t
        
        # === ATTACKERS ACT ===
        for atk in attackers:
            action, i = atk.step(G)
            node_id = G.nodes[i]['node_id']
            
            # === UPDATE FEATURES (SAFE - all node_ids exist in X_unscaled_current) ===
            if action == 'exploit':
                if 'exploit_attempts' in X_unscaled_current.columns:
                    X_unscaled_current.loc[node_id, 'exploit_attempts'] += 1
                
                if G.nodes[i]['is_honeypot']:
                    stats['honeypot_hits'] += 1
                    # === LOCAL TOPOLOGY UPDATE + SYNC ===
                    X_unscaled_current, node_id_to_graph_map, graph_to_node_id_map, node_ids = update_topology_on_honeypot_hit(
                        G, i, X_unscaled_current, node_id_to_graph_map, graph_to_node_id_map, node_ids
                    )
                elif node_id not in compromised:
                    compromised.add(node_id)
                    stats['real_hits'] += 1
                    # Optional: Spawn new attacker (worm-like spread)
                    if random.random() < 0.1:
                        attackers.append(RandomAttacker(i))
            else:  # scan
                if 'scan_count' in X_unscaled_current.columns:
                    X_unscaled_current.loc[node_id, 'scan_count'] += 1

        # === CONTROLLER ACTS EVERY 3 STEPS ===
        if t % 3 == 0:
            # Re-scale ALL current features (including new nodes)
            X_scaled_current = scaler.transform(X_unscaled_current.values)
            
            # Compute AE scores for ALL nodes
            ae_scores = ae_reconstruction_error(ae_model, X_scaled_current)
            
            # === SELECT TOP ANOMALOUS NODES ===
            # Map scores to graph indices (handles new nodes!)
            node_score_dict = {}
            for idx, node_id in enumerate(node_ids):
                graph_idx = node_id_to_graph_map.get(node_id, -1)
                if graph_idx != -1:
                    node_score_dict[graph_idx] = ae_scores[idx]
            
            top_nodes_by_score = sorted(node_score_dict.items(), key=lambda x: x[1], reverse=True)
            top_graph_nodes = [i for i, s in top_nodes_by_score[:topk_spawn] if s > ae_threshold]
            
            # Deploy honeypots
            controller.apply(G, top_graph_nodes)
            
            if top_graph_nodes:
                print(f"[t={t}] Deployed {len(top_graph_nodes)} HPs | Max AE: {max(ae_scores):.3f} | HP hits: {stats['honeypot_hits']} | Real: {stats['real_hits']}")

    stats['cost'] = controller.cost
    stats['compromised_nodes'] = len(compromised)
    stats['compromised_list'] = list(compromised)
    stats['final_nodes'] = len(G.nodes())

    print("\n=== FINAL STATS ===")
    print(stats)
    return stats


# ---------------------------
# EXECUTION BLOCK
# ---------------------------
if __name__ == '__main__':
    # -----------------------------
    # Run anomaly detection pipeline
    # -----------------------------
    try:
        out_df, metrics, ae_model, if_model, lof_model, features_test_df, scaler, ae_threshold = run_all(save_outputs=True)
    except Exception as e:
        print(f"\nFATAL ERROR during ML Pipeline: {e}")
        exit()
    
    feat_names = features_test_df.columns.tolist()
    print(f"\nFeatures available for simulation: {feat_names}")
    
    # -----------------------------
    # Run adaptive honeypot simulation (AE-only + LOCAL TOPOLOGY)
    # -----------------------------
    print("\n Starting Adaptive Honeypot Simulation (AE-only + Local Topology)...")
    stats = run_honeypot_sim_adaptive(
        features_test_df=features_test_df,
        ae_model=ae_model,
        if_clf=if_model,
        lof_clf=lof_model,
        scaler=scaler,
        feat_names=feat_names,
        ae_threshold=ae_threshold,  # Dynamic 95th percentile!
        topk_spawn=8,
        max_timesteps=MAX_TIMESTEPS
    )

    print("\n SIMULATION COMPLETE!")
    print(f"Effectiveness: {stats['honeypot_hits'] / max(stats['honeypot_hits'] + stats['real_hits'], 1):.1%} honeypots hit")
    print(f"Cost: {stats['cost']:.1f} | Compromised: {stats['compromised_nodes']}")