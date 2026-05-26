"""
ContexGuard - Full Evaluation Script for Paper Revision
Generates: confusion matrix, precision/recall/F1/AUC, multi-model comparison,
ablation study, and statistical validation with confidence intervals.

Uses realistic synthetic data with overlapping decision boundaries.
"""
import os, sys, time, json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)


def generate_realistic_data(num_samples=150000, seed=42):
    """
    Generate realistic contextual data with overlapping boundaries.
    In real edge deployments, attack patterns are NOT perfectly separable:
    - Some legitimate requests come from unusual locations
    - Some attacks use stolen credentials with high device scores
    - Boundary cases create natural classification difficulty
    """
    rng = np.random.RandomState(seed)
    X = []
    y = []

    for _ in range(num_samples):
        is_attack = rng.rand() < 0.15

        if is_attack:
            attack_type = rng.choice(['clear', 'subtle', 'sophisticated', 'boundary'])

            if attack_type == 'clear':  # ~40% of attacks: obvious
                device_score = rng.uniform(0.0, 0.35)
                loc_trust = 0.0
                time_anomaly = rng.uniform(0.6, 1.0)
                net_anomaly = rng.uniform(0.65, 1.0)
            elif attack_type == 'subtle':  # ~25% of attacks: moderate signals
                device_score = rng.uniform(0.3, 0.6)
                loc_trust = rng.choice([0.0, 1.0], p=[0.7, 0.3])
                time_anomaly = rng.uniform(0.35, 0.7)
                net_anomaly = rng.uniform(0.4, 0.75)
            elif attack_type == 'sophisticated':  # ~20% of attacks: mimics legitimate
                device_score = rng.uniform(0.6, 0.85)
                loc_trust = 1.0  # spoofed location
                time_anomaly = rng.uniform(0.2, 0.5)
                net_anomaly = rng.uniform(0.55, 0.85)
            else:  # ~15% of attacks: boundary/ambiguous
                device_score = rng.uniform(0.45, 0.7)
                loc_trust = rng.choice([0.0, 1.0])
                time_anomaly = rng.uniform(0.25, 0.55)
                net_anomaly = rng.uniform(0.3, 0.6)
            label = 1
        else:
            legit_type = rng.choice(['normal', 'unusual_but_legit', 'boundary'])

            if legit_type == 'normal':  # ~75% of legit: clear
                device_score = rng.uniform(0.75, 1.0)
                loc_trust = 1.0
                time_anomaly = rng.uniform(0.0, 0.2)
                net_anomaly = rng.uniform(0.0, 0.15)
            elif legit_type == 'unusual_but_legit':  # ~15%: traveling employee etc.
                device_score = rng.uniform(0.6, 0.9)
                loc_trust = 0.0  # working remotely
                time_anomaly = rng.uniform(0.1, 0.45)
                net_anomaly = rng.uniform(0.05, 0.3)
            else:  # ~10%: boundary legitimate
                device_score = rng.uniform(0.5, 0.75)
                loc_trust = rng.choice([0.0, 1.0])
                time_anomaly = rng.uniform(0.15, 0.4)
                net_anomaly = rng.uniform(0.1, 0.4)
            label = 0

        # Add realistic sensor noise
        device_score = np.clip(device_score + rng.normal(0, 0.08), 0, 1)
        time_anomaly = np.clip(time_anomaly + rng.normal(0, 0.06), 0, 1)
        net_anomaly = np.clip(net_anomaly + rng.normal(0, 0.07), 0, 1)

        X.append([device_score, loc_trust, time_anomaly, net_anomaly])
        y.append(label)

    return np.array(X), np.array(y)


def run_full_evaluation():
    results = {}

    # ===== 1. Generate Data =====
    print("=" * 70)
    print("PHASE 1: Data Generation (150,000 samples)")
    print("=" * 70)
    X, y = generate_realistic_data(num_samples=150000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = ["Device Security Score", "Location Trust",
                     "Time Anomaly", "Network Anomaly"]

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution (Train): Permit={sum(y_train==0)}, Deny={sum(y_train==1)}")
    print(f"Class distribution (Test):  Permit={sum(y_test==0)}, Deny={sum(y_test==1)}")

    results['dataset'] = {
        'total_samples': int(len(X)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_permit': int(sum(y_train == 0)),
        'train_deny': int(sum(y_train == 1)),
        'test_permit': int(sum(y_test == 0)),
        'test_deny': int(sum(y_test == 1)),
    }

    # ===== 2. Multi-Model Comparison =====
    print("\n" + "=" * 70)
    print("PHASE 2: Multi-Model Comparative Analysis")
    print("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=42
        ),
    }

    model_results = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")

        # Train
        t_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = (time.perf_counter() - t_start) * 1000

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Inference latency (average of 5,000 single predictions)
        latencies = []
        for i in range(5000):
            idx = i % len(X_test)
            t0 = time.perf_counter()
            model.predict(X_test[idx:idx+1])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        model_results[name] = {
            'accuracy': float(round(acc, 4)),
            'precision': float(round(prec, 4)),
            'recall': float(round(rec, 4)),
            'f1_score': float(round(f1, 4)),
            'auc_roc': float(round(auc, 4)),
            'confusion_matrix': cm.tolist(),
            'avg_inference_latency_ms': float(round(avg_latency, 4)),
            'std_inference_latency_ms': float(round(std_latency, 4)),
            'training_time_ms': float(round(train_time, 2)),
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}, TP={cm[1][1]}")
        print(f"  Avg Inference Latency: {avg_latency:.4f} ms (+/-{std_latency:.4f})")
        print(f"  Training Time: {train_time:.2f} ms")
        print(classification_report(y_test, y_pred, target_names=['Permit', 'Deny']))

    results['model_comparison'] = model_results

    # ===== 3. Ablation Study (Feature Removal) =====
    print("\n" + "=" * 70)
    print("PHASE 3: Ablation Study - Feature Removal Impact on Random Forest")
    print("=" * 70)

    ablation_results = {}

    # Full model baseline
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_full.fit(X_train, y_train)
    y_pred_full = rf_full.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred_full)
    baseline_acc = accuracy_score(y_test, y_pred_full)
    ablation_results["All Features"] = {
        'f1_score': float(round(baseline_f1, 4)),
        'accuracy': float(round(baseline_acc, 4)),
        'features_used': feature_names,
    }
    print(f"  Baseline (All Features): F1={baseline_f1:.4f}, Acc={baseline_acc:.4f}")

    # Remove one feature at a time
    for i, fname in enumerate(feature_names):
        mask = [j for j in range(len(feature_names)) if j != i]
        X_train_abl = X_train[:, mask]
        X_test_abl = X_test[:, mask]

        rf_abl = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_abl.fit(X_train_abl, y_train)
        y_pred_abl = rf_abl.predict(X_test_abl)

        abl_f1 = f1_score(y_test, y_pred_abl)
        abl_acc = accuracy_score(y_test, y_pred_abl)
        delta_f1 = abl_f1 - baseline_f1

        ablation_results[f"Without {fname}"] = {
            'f1_score': float(round(abl_f1, 4)),
            'accuracy': float(round(abl_acc, 4)),
            'delta_f1': float(round(delta_f1, 4)),
            'features_used': [f for j, f in enumerate(feature_names) if j != i],
        }
        print(f"  Without '{fname}': F1={abl_f1:.4f} (delta={delta_f1:+.4f}), Acc={abl_acc:.4f}")

    # Feature importances from the full RF model
    importances = rf_full.feature_importances_
    ablation_results["feature_importances"] = {
        fname: float(round(imp, 4)) for fname, imp in zip(feature_names, importances)
    }
    print(f"\n  Feature Importances: {dict(zip(feature_names, importances.round(4)))}")

    results['ablation_study'] = ablation_results

    # ===== 4. Statistical Validation (10 Independent Runs) =====
    print("\n" + "=" * 70)
    print("PHASE 4: Statistical Validation - 10 Independent Runs")
    print("=" * 70)

    run_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for run in range(10):
        seed = run * 7 + 1
        X_r, y_r = generate_realistic_data(num_samples=150000, seed=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_r, y_r, test_size=0.2, random_state=seed, stratify=y_r
        )
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
        rf.fit(X_tr, y_tr)
        yp = rf.predict(X_te)
        yprob = rf.predict_proba(X_te)[:, 1]

        run_metrics['accuracy'].append(accuracy_score(y_te, yp))
        run_metrics['precision'].append(precision_score(y_te, yp))
        run_metrics['recall'].append(recall_score(y_te, yp))
        run_metrics['f1'].append(f1_score(y_te, yp))
        run_metrics['auc'].append(roc_auc_score(y_te, yprob))

        print(f"  Run {run+1:2d}: Acc={run_metrics['accuracy'][-1]:.4f}, "
              f"P={run_metrics['precision'][-1]:.4f}, "
              f"R={run_metrics['recall'][-1]:.4f}, "
              f"F1={run_metrics['f1'][-1]:.4f}, "
              f"AUC={run_metrics['auc'][-1]:.4f}")

    stat_results = {}
    for metric_name, values in run_metrics.items():
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        ci_95 = float(1.96 * std / np.sqrt(len(arr)))
        stat_results[metric_name] = {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'ci_95_lower': round(mean - ci_95, 4),
            'ci_95_upper': round(mean + ci_95, 4),
        }
        print(f"  {metric_name:12s}: {mean:.4f} +/- {std:.4f} "
              f"(95%% CI: [{mean-ci_95:.4f}, {mean+ci_95:.4f}])")

    results['statistical_validation'] = stat_results

    # ===== 5. Security-Focused Evaluation =====
    print("\n" + "=" * 70)
    print("PHASE 5: Security-Focused Evaluation - Per-Attack-Type Detection")
    print("=" * 70)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    rng = np.random.RandomState(99)
    attack_scenarios = {
        "Geo-Velocity Violation": {
            "device_score": rng.uniform(0.7, 0.9, 1000),
            "loc_trust": np.zeros(1000),
            "time_anomaly": rng.uniform(0.7, 1.0, 1000),
            "net_anomaly": rng.uniform(0.0, 0.3, 1000),
        },
        "Device Posture Degradation": {
            "device_score": rng.uniform(0.0, 0.3, 1000),
            "loc_trust": np.ones(1000),
            "time_anomaly": rng.uniform(0.0, 0.2, 1000),
            "net_anomaly": rng.uniform(0.0, 0.2, 1000),
        },
        "Network Anomaly (DoS)": {
            "device_score": rng.uniform(0.7, 1.0, 1000),
            "loc_trust": np.ones(1000),
            "time_anomaly": rng.uniform(0.0, 0.2, 1000),
            "net_anomaly": rng.uniform(0.8, 1.0, 1000),
        },
        "Combined Multi-Vector": {
            "device_score": rng.uniform(0.0, 0.3, 1000),
            "loc_trust": np.zeros(1000),
            "time_anomaly": rng.uniform(0.7, 1.0, 1000),
            "net_anomaly": rng.uniform(0.8, 1.0, 1000),
        },
        "Sophisticated Mimicry": {
            "device_score": rng.uniform(0.6, 0.85, 1000),
            "loc_trust": np.ones(1000),
            "time_anomaly": rng.uniform(0.2, 0.5, 1000),
            "net_anomaly": rng.uniform(0.55, 0.85, 1000),
        },
    }

    security_results = {}
    for scenario_name, features in attack_scenarios.items():
        X_attack = np.column_stack([
            features["device_score"],
            features["loc_trust"],
            features["time_anomaly"],
            features["net_anomaly"],
        ])
        predictions = rf_model.predict(X_attack)
        detection_rate = sum(predictions == 1) / len(predictions)
        security_results[scenario_name] = {
            'detection_rate': float(round(detection_rate, 4)),
            'total_samples': int(len(predictions)),
            'detected': int(sum(predictions == 1)),
            'missed': int(sum(predictions == 0)),
        }
        print(f"  {scenario_name:40s}: Detection = {detection_rate:.2%} "
              f"({sum(predictions==1)}/{len(predictions)})")

    results['security_evaluation'] = security_results

    # ===== Save All Results =====
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 70}")
    print(f"All results saved to: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_full_evaluation()
