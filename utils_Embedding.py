
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from functools import partial
from sklearn.pipeline import Pipeline
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import optuna.visualization as vis
############################################################################################
#STATIC EMBEDDING
############################################################################################
# Calculates correlation matrices
def static_correlation_matrices(data_list):
    """ This function calculate the static correlation matrices
    of the subjects in data_list.
    Parameters:
    data_list: list of dicts [dict(),..] each dict contains the singnals of the subjects
    returns:
    corr_matrices: list of numpy arrays
    """
    corr_matrices = []
    for subj in data_list:
        bold = subj["bold"]
        corr_matrix = np.corrcoef(bold.T)
        corr_matrices.append(corr_matrix)
    return corr_matrices

# Function that search for the best params of UMAP 
def objective(trial, X, labels, splits,seed):
    # sugerencias de hiperparámetros
    n_neighbors = trial.suggest_int("n_neighbors", 5, 98)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.9)
    n_components = trial.suggest_int("n_components", 2, 50)
    metric = trial.suggest_categorical("metric", ["cosine", "correlation", "euclidean"])

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed
    )
    # Z = reducer.fit_transform(X_scaled)

    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('umap', reducer),
        ('lr', clf)
    ])
    
    #acc = cross_val_score(clf, Z, labels, cv=cv, scoring="accuracy").mean()
    acc = cross_val_score(pipe, X, labels, cv=splits, scoring="accuracy").mean()

    return acc


# plot the borders beetween groups in the visualization of PCA
def plot_lr_boundary_on_2d(Z2, y, C=1.0, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    clf2d = LogisticRegression(C=C, max_iter=1000)
    clf2d.fit(Z2, y)

    x_min, x_max = Z2[:,0].min()-0.5, Z2[:,0].max()+0.5
    y_min, y_max = Z2[:,1].min()-0.5, Z2[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    proba = clf2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    cf = ax.contourf(xx, yy, proba, levels=np.linspace(0,1,21),
                     cmap="coolwarm", alpha=0.30)
    ax.contour(xx, yy, proba, levels=[0.5], colors='k', linewidths=2)

    return clf2d, cf

def optimal_params_UMAP(X, labels, splits,components= None, study=True,seed=None):
    """Search the optimal parameters of UMAP and show the embeddings
    parameters:
    X: data BOLD
    labels: array size len(dataset)
    seed: int
    returns:
    embedding: the UMAP model
    reducer: the UMAP model
    labels: array size len(dataset)
    params: dict with the optimal parameters

    """
    np.random.seed(seed)
    random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if study:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study_obj = optuna.create_study(direction="maximize", sampler=sampler)

        objective_partial = partial(objective, X=X, labels=labels, splits=splits,seed=seed)
        study_obj.optimize(objective_partial, n_trials=100)

        vis.plot_optimization_history(study_obj).show()
        vis.plot_param_importances(study_obj).show()

        # params = study.best_params
        sorted_trials = sorted(study_obj.trials, key=lambda t: t.value, reverse=True)

        # top 3
        top3 = sorted_trials[:3]

        for i, trial in enumerate(top3):
            params = trial.params
            reducer = umap.UMAP(
              n_neighbors=params["n_neighbors"],
              min_dist=params["min_dist"],
              n_components=params["n_components"],
              metric=params["metric"],
              random_state=seed
              )
            #embedding = reducer.fit_transform(X_scaled) # Unsupervised
            #Just for visualization
            #embedding = reducer.fit_transform(X, y=labels) # Supervised 
            #
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('umap', reducer),
                ('lr', LogisticRegression(max_iter=1000))
            ])
            # Classification metrics
            #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            #clf_full = LogisticRegression(max_iter=1000)
            acc_full = cross_val_score(pipe, X, labels, cv=splits, scoring='accuracy').mean()
            auc_full = cross_val_score(pipe, X, labels, cv=splits, scoring='roc_auc').mean()

            # Visualization
            scaler_viz = StandardScaler()
            X_viz = scaler_viz.fit_transform(X)

            reducer_viz = umap.UMAP(
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            n_components=2,
            metric=params["metric"],
            random_state=seed
            )

            Z_viz = reducer_viz.fit_transform(X_viz)#, y=labels)

            #pca = PCA(n_components=2)
            #Z_pca = pca.fit_transform(embedding)

            fig, ax = plt.subplots(figsize=(7,6))

            clf2d, cf = plot_lr_boundary_on_2d(Z_viz, labels, C=1.0, ax=ax)
            colors = ['steelblue' if l==0 else 'tomato' for l in labels]
            plt.scatter(Z_viz[:,0], Z_viz[:,1], c=colors, s=60, alpha=0.8, edgecolor='k')

            plt.title(f"TOP {i+1} embedding ({params['n_components']}D UMAP) Acc={acc_full:.3f} AUC={auc_full:.3f}")
            #plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
            #plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            text_str = (
              f"n_neighbors = {params['n_neighbors']},  "
              f"min_dist = {params['min_dist']:.2f},  "
              f"n_components = {params['n_components']},  "
              f"metric = {params['metric']}"
              )

            plt.figtext(0.5, -0.05, text_str, wrap=True, ha="center", fontsize=10, color="dimgray")
            plt.tight_layout()
            plt.show()

        best_params = study_obj.best_trial.params

        # best_reducer = umap.UMAP(
        #       n_neighbors=params["n_neighbors"],
        #       min_dist=params["min_dist"],
        #       n_components=params["n_components"],
        #       metric=params["metric"],
        #       random_state=seed
        #       )
        # best_embedding = best_reducer.fit_transform(X, y=labels)
    else:
        print("Study disable")
        best_embedding, best_reducer, best_params = None, None, None

    return labels, best_params
############################################################################################
#DYNAMIC EMBEDDING
############################################################################################

# Dynamic Embedding
def compute_subject_TP_from_Xi(Xi, Gamma, indices, K, eps=1e-8):
    """
    Xi     : array (T_total-1, K, K)
    Gamma  : array (T_total, K)
    indices: array (n_subj, 2)
    K      : number of states

    Returns
    -------
    A_subj : array (n_subj, K, K)
             Transition probability matrix per subject
    """

    n_subj = indices.shape[0]
    A_subj = np.zeros((n_subj, K, K))

    for s in range(n_subj):
        t0, t1 = indices[s]

        # Xi goes from t to t+1, so last timepoint excluded
        Xi_s = Xi[t0:t1-1]              # (T_s-1, K, K)
        Gamma_s = Gamma[t0:t1-1]        # (T_s-1, K)

        # Expected transitions
        num = Xi_s.sum(axis=0)          # (K, K)
        den = Gamma_s.sum(axis=0)[:, None]  # (K, 1)

        A_s = num / (den + eps)
        A_subj[s] = A_s

    return A_subj

def build_dynamic_embedding(fo, dwell, switch_rate, A):
    """
    Build dynamic embeddings for each subject.

    Parameters
    fo : array (n_subj, K), Fractional occupancies.
    dwell : array (n_subj, K) mean Dwell times.
    switch_rate : array (n_subj,K) Switching rate per state for subject.
    A : array (n_subj, K, K) Transition matrices for subject

    Returns
    -------
    X_dyn : array (n_subj, dim_embedding)
        dynamic embedding for subject
    """

    n_subj, K = fo.shape
    embeddings = []

    mask = ~np.eye(K, dtype=bool)

    for s in range(n_subj):
        # 1. Fractional Occupancy
        fo_s = fo[s]                   # (K,)

        # 2. Dwell time por estado
        dwell_s = dwell[s]             # (K,)

        # 3. Switching por estado (ya es vector de longitud K)
        sw_s = switch_rate[s]          # (K,)

        # 4. Matriz de transición vectorizada (upper triangular sin diagonal)
        A_s = A[s]      
        A_flat = A_s[mask]  # (K*K - K,)               # (K, K)

        # Concatenar todas las features
        feat_s = np.concatenate([fo_s, dwell_s, sw_s, A_flat])
        embeddings.append(feat_s)

    X_dyn = np.vstack(embeddings)
    return X_dyn

def dyn_embedding_from_hmm(BOLD, K, preproc, glhmm, utils, seed=123, covtype="full"):
    """
    BOLD: array (n_subj, T, R)
    preproc: preproc object (preproc.preprocess_data)
    glhmm: module glhmm (glhmm.glhmm)
    utils: utils 
    """
    from glhmm.auxiliary import make_indices_from_T

    n_subj, T, R = BOLD.shape

    # concatenation (time axis)
    Y_concat = np.concatenate([BOLD[i] for i in range(n_subj)], axis=0)  # (T_total, R)
    indices = make_indices_from_T([T] * n_subj)

    # preprocess 
    Y_pre, _, log = preproc.preprocess_data(Y_concat, indices)

    # train HMM
    hmm = glhmm.glhmm(model_beta='no', K=K, covtype=covtype, preproclogY=log)
    np.random.seed(seed)
    Gamma, Xi, FE = hmm.train(X=None, Y=Y_pre, indices=indices)

    # fraction occ
    FO = utils.get_FO(Gamma, indices=indices)

    # Switching rate
    try:
        SR = utils.get_switching_rate(Gamma, indices=indices)
    except TypeError:
        SR = utils.get_switching_rate(Gamma, indices)

    # Dwell time
    vpath_prob = hmm.decode(X=None, Y=Y_pre, indices=indices, viterbi=True)
    
    LTmean, LTmed, LTmax = utils.get_life_times(vpath_prob, indices)

    # A 
    A = compute_subject_TP_from_Xi(Xi, Gamma, indices, K)

    # build embedding
    X_dyn = build_dynamic_embedding(FO, LTmean, SR, A)

    return X_dyn, FE, hmm

def evaluate_embedding_lr(X, labels, splits, scoring="accuracy", C=1.0, seed=0):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000))#, C=C, solver="lbfgs"))
    ])
    scores = cross_val_score(pipe, X, labels, cv=splits, scoring=scoring)
    return scores  # returns fold to fold

####################################################
def _fe_to_scalar(FE):
    if FE is None:
        return np.nan
    if isinstance(FE, (list, tuple, np.ndarray)):
        FE = np.asarray(FE).ravel()
        return float(FE[-1])
    return float(FE)

def _kneedle_elbow(Ks, y):
    Ks = np.asarray(Ks, dtype=float)
    y = np.asarray(y, dtype=float)

    # Normalization
    x_norm = (Ks - Ks.min()) / (Ks.max() - Ks.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 
    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    dx, dy = (x1 - x0), (y1 - y0)

   
    # |dy*x - dx*y + x1*y0 - y1*x0| / sqrt(dx^2+dy^2)
    denom = np.sqrt(dx*dx + dy*dy) + 1e-12
    dist = np.abs(dy * x_norm - dx * y_norm + (x1 * y0 - y1 * x0)) / denom

    idx = int(np.argmax(dist))
    return int(Ks[idx])


def fit_hmm_get_fe_and_active_states(BOLD, K, preproc, glhmm, utils, seed=0, model_beta="no", covtype="full", verbose=False):
    """
    Train glhmm and returns:
      FE (float),
      active_states (int),
      FO_mean (K,), FO_min (float)
    """
    from glhmm.auxiliary import make_indices_from_T

    rng = np.random.RandomState(seed)

    n_subj, T, R = BOLD.shape
    X_concat = np.concatenate([BOLD[i] for i in range(n_subj)], axis=0)  # (n_subj*T, R)
    Ts = [T] * n_subj
    indices = make_indices_from_T(Ts)

    # preprocess 
    Y, _, log = preproc.preprocess_data(X_concat, indices)

    hmm = glhmm.glhmm(model_beta=model_beta, K=K, covtype=covtype, preproclogY=log)

    np.random.seed(seed)  
    Gamma, Xi, FE = hmm.train(X=None, Y=Y, indices=indices)

    FE_scalar = _fe_to_scalar(FE)

    # FO per subject
    FO = utils.get_FO(Gamma, indices=indices)   # n_subj, K
    FO_mean = np.asarray(FO).mean(axis=0)       # K
    FO_min = float(np.min(FO_mean))

    if verbose:
        print(f"K={K} seed={seed} FE={FE_scalar:.3f} FO_min={FO_min:.6f}")

    return FE_scalar, FO_mean, FO_min


def select_K_by_FE_elbow_no_inactive(
    BOLD,
    K_list,
    preproc,
    glhmm,
    utils,
    seeds=(0, 1, 2),
    fo_min=1e-3,          # threshold
    require_all_active=True,  
    verbose=True
):
    """
    returns
      best_K,
      summary (list dict per K),
      chosen_reason (str)
    """

    summary = []
    for K in K_list:
        FE_vals = []
        FOmins = []
        
        for sd in seeds:
            FE_sc, FO_mean, FO_min_val = fit_hmm_get_fe_and_active_states(
                BOLD=BOLD, K=K, preproc=preproc, glhmm=glhmm, utils=utils,
                seed=sd, verbose=False
            )
            FE_vals.append(FE_sc)
            FOmins.append(FO_min_val)

        FE_vals = np.asarray(FE_vals, dtype=float)
        FOmins  = np.asarray(FOmins, dtype=float)

        FE_mean = float(np.nanmean(FE_vals))
        FE_std  = float(np.nanstd(FE_vals))
        FOmin_mean = float(np.nanmean(FOmins))
        FOmin_std  = float(np.nanstd(FOmins))

        # without inactive states
        if require_all_active:
            ok_active = (FOmin_mean > fo_min)  # 
        else:
            # 
            ok_active = (FOmin_mean > fo_min/10)

        summary.append({
            "K": K,
            "FE_mean": FE_mean,
            "FE_std": FE_std,
            "FOmin_mean": FOmin_mean,
            "FOmin_std": FOmin_std,
            "ok_active": bool(ok_active)
        })

        if verbose:
            print(f"K={K:2d} | FE={FE_mean:.3f}±{FE_std:.3f} | FOmin={FOmin_mean:.6f}±{FOmin_std:.6f} | active_ok={ok_active}")

    # Filter valid K
    valid = [d for d in summary if d["ok_active"]]
    if len(valid) == 0:
  
        Ks_all = [d["K"] for d in summary]
        FEs_all = [d["FE_mean"] for d in summary]
        best_K = _kneedle_elbow(Ks_all, FEs_all)
        reason = f"Elbow (FE) over all K (no K passed the active-state filter with fo_min={fo_min})."
        if verbose:
            print("WARNING:", reason)
        return best_K, summary, reason

    # Apply elbow on valid Ks
    Ks = [d["K"] for d in valid]
    FEs = [d["FE_mean"] for d in valid]
    elbow_K = _kneedle_elbow(Ks, FEs)

    candidates = [d["K"] for d in valid if d["K"] >= elbow_K]
    if len(candidates) == 0:
        best_K = elbow_K
    else:
        best_K = int(min(candidates))

    reason = f"Elbow (FE) on valid K + parsimony (choose the smallest K ≥ elbow). elbow={elbow_K}, chosen={best_K}"
    if verbose:
        print("Chosen:", reason)

    #Visualization
    Ks = [d["K"] for d in summary]
    FE = [d["FE_mean"] for d in summary]
    FEstd = [d["FE_std"] for d in summary]
    ok = [d["ok_active"] for d in summary]

    plt.figure()
    plt.errorbar(Ks, FE, yerr=FEstd, marker="o", linestyle="-", capsize=3)
    for k, fe, isok in zip(Ks, FE, ok):
        if not isok:
            plt.scatter([k], [fe], marker="x", s=80)  # invalids

    plt.axvline(best_K, linestyle="--")
    plt.xlabel("K")
    plt.ylabel("Final Free Energy (mean ± std across seeds)")
    plt.title("Selecting K by FE elbow + no inactive states")
    plt.show()


    return best_K, summary, reason