#!/usr/bin/env python3
"""
Analyze root cause separation using PROPAGATION ACCURACY (not purity).

1. Calculate baseline propagation accuracy at exit code pattern level
2. Find optimal k using elbow method (WITHOUT labels - unsupervised)
3. Cluster with optimal k
4. Propagate root cause labels from medoids to all points
5. Measure propagation accuracy
6. Compare baseline vs k-means clustering
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

PACKAGE_ROOT = Path(__file__).parent.parent.parent
LABELS_FILE = PACKAGE_ROOT / "data" / "labels" / "manual_labels_with_root_cause.json"
FEATURES_FILE = PACKAGE_ROOT / "data" / "feature_vectors.json"
OUTPUT_FILE = Path(__file__).parent / "results" / "root_cause_propagation_analysis.json"
REPORT_FILE = Path(__file__).parent / "results" / "root_cause_propagation_report.md"
PLOT_FILE = Path(__file__).parent / "results" / "root_cause_propagation_comparison.png"

# TF-IDF parameters
TFIDF_PARAMS = {
    "max_features": 1100,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.80,
}

# K-means parameters
N_REPEATS = 5  # Number of repetitions for each k value
MAX_K = 20  # Maximum k to test per pattern


def load_labels():
    """Load labels with root causes."""
    with open(LABELS_FILE) as f:
        return json.load(f)


def load_features():
    """Load feature vectors."""
    with open(FEATURES_FILE) as f:
        data = json.load(f)
    return data.get('feature_vectors', data)


def find_centroid_medoid(tfidf_matrix, cluster_indices):
    """Find the medoid (point closest to centroid) for a cluster."""
    if len(cluster_indices) == 0:
        return None
    if len(cluster_indices) == 1:
        return cluster_indices[0]
    
    cluster_vectors = tfidf_matrix[cluster_indices]
    if hasattr(cluster_vectors, 'toarray'):
        cluster_vectors = cluster_vectors.toarray()
    cluster_vectors = np.asarray(cluster_vectors)
    
    centroid = np.mean(cluster_vectors, axis=0)
    centroid_reshaped = np.asarray(centroid).reshape(1, -1)
    distances = cosine_distances(centroid_reshaped, cluster_vectors)[0]
    medoid_idx = np.argmin(distances)
    
    return cluster_indices[medoid_idx]


def calculate_propagation_accuracy(cluster_labels, test_ids, true_labels_dict, tfidf_matrix):
    """
    Calculate propagation accuracy: propagate medoid's label to all points in cluster.
    
    Args:
        cluster_labels: List of cluster assignments
        test_ids: List of test IDs corresponding to cluster_labels
        true_labels_dict: Dict mapping test_id -> root_cause
        tfidf_matrix: TF-IDF matrix for finding medoids
    
    Returns:
        accuracy: Propagation accuracy
        propagated_labels: Dict of {test_id: propagated_label}
    """
    if len(cluster_labels) == 0:
        return 0.0, {}
    
    # Group by cluster
    clusters = defaultdict(list)
    for idx, (cluster_id, test_id) in enumerate(zip(cluster_labels, test_ids)):
        clusters[cluster_id].append((idx, test_id))
    
    propagated_labels = {}
    all_true_labels = {}
    all_propagated_labels = []
    all_true_labels_list = []
    
    for cluster_id, points in clusters.items():
        if len(points) == 0:
            continue
        
        # Get indices for this cluster
        cluster_indices = [idx for idx, _ in points]
        
        # Find medoid
        medoid_idx = find_centroid_medoid(tfidf_matrix, cluster_indices)
        if medoid_idx is None:
            continue
        
        # Get medoid's test_id and label
        medoid_test_id = test_ids[medoid_idx]
        medoid_label = true_labels_dict.get(medoid_test_id, None)
        
        if medoid_label is None:
            continue
        
        # Propagate medoid's label to all points in cluster
        for idx, test_id in points:
            propagated_labels[test_id] = medoid_label
            true_label = true_labels_dict.get(test_id, None)
            
            if true_label is not None:
                all_propagated_labels.append(medoid_label)
                all_true_labels_list.append(true_label)
    
    # Calculate accuracy
    if len(all_propagated_labels) == 0:
        return 0.0, propagated_labels
    
    correct = sum(1 for p, t in zip(all_propagated_labels, all_true_labels_list) if p == t)
    accuracy = correct / len(all_propagated_labels) if len(all_propagated_labels) > 0 else 0.0
    
    return accuracy, propagated_labels


def calculate_baseline_propagation_accuracy(labels):
    """
    Calculate baseline propagation accuracy at exit code pattern level.
    Each pattern is treated as one cluster - propagate medoid's label to all in pattern.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pathlib import Path
    
    ORGANIZED_DIR = Path(__file__).parent.parent / "organized_9_21"
    features = load_features()
    
    pattern_results = {}
    
    # Group by pattern
    for test_id, label_data in labels.items():
        pattern = label_data.get('pattern', 'unknown')
        root_cause = label_data.get('root_cause', 'UNKNOWN')
        
        if pattern not in pattern_results:
            pattern_results[pattern] = {
                'test_ids': [],
                'root_causes': [],
                'texts': []
            }
        
        pattern_results[pattern]['test_ids'].append(test_id)
        pattern_results[pattern]['root_causes'].append(root_cause)
        
        # Get text feature
        if test_id in features:
            pattern_results[pattern]['texts'].append(features[test_id].get('text', ''))
        else:
            pattern_results[pattern]['texts'].append('')
    
    baseline_accuracies = {}
    
    for pattern, data in pattern_results.items():
        if len(data['test_ids']) < 1:
            continue
        
        if len(data['test_ids']) == 1:
            # Single point - perfect accuracy
            baseline_accuracies[pattern] = {
                'accuracy': 1.0,
                'num_findings': 1,
                'unique_root_causes': 1
            }
            continue
        
        # Create TF-IDF matrix
        try:
            vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
            tfidf_matrix = vectorizer.fit_transform(data['texts'])
        except:
            baseline_accuracies[pattern] = {
                'accuracy': 0.0,
                'num_findings': len(data['test_ids']),
                'unique_root_causes': len(set(data['root_causes']))
            }
            continue
        
        # All points in same "cluster" (pattern)
        cluster_labels = [0] * len(data['test_ids'])
        true_labels_dict = {tid: rc for tid, rc in zip(data['test_ids'], data['root_causes'])}
        
        accuracy, _ = calculate_propagation_accuracy(
            cluster_labels, data['test_ids'], true_labels_dict, tfidf_matrix
        )
        
        baseline_accuracies[pattern] = {
            'accuracy': accuracy,
            'num_findings': len(data['test_ids']),
            'unique_root_causes': len(set(data['root_causes'])),
            'root_cause_distribution': dict(Counter(data['root_causes']))
        }
    
    return baseline_accuracies


def find_optimal_k_elbow(texts, max_k=None):
    """
    Find optimal k using elbow method based on silhouette score and WCSS.
    NO LABELS USED - purely unsupervised.
    
    Returns:
        optimal_k: Optimal k value
        results: Dict with k values and their metrics
    """
    if len(texts) < 2:
        return 1, {}
    
    if max_k is None:
        max_k = min(MAX_K, len(texts) - 1)
    else:
        max_k = min(max_k, len(texts) - 1, MAX_K)
    
    if max_k < 1:
        return 1, {}
    
    # Create TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
        return 1, {}
    
    results = {}
    best_k = 1
    best_score = -1
    
    for k in range(1, max_k + 1):
        if k >= len(texts):
            break
        
        sil_scores = []
        wcss_scores = []
        
        for seed in range(N_REPEATS):
            try:
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Calculate silhouette score
                if k > 1 and len(set(cluster_labels)) > 1:
                    sil_score = silhouette_score(tfidf_matrix, cluster_labels)
                    sil_scores.append(sil_score)
                
                # Calculate WCSS (within-cluster sum of squares)
                wcss = kmeans.inertia_
                wcss_scores.append(wcss)
                
            except Exception as e:
                continue
        
        if not sil_scores and k > 1:
            continue
        
        avg_sil = np.mean(sil_scores) if sil_scores else 0.0
        avg_wcss = np.mean(wcss_scores) if wcss_scores else 0.0
        
        results[k] = {
            'silhouette_score': avg_sil,
            'wcss': avg_wcss
        }
        
        # Use silhouette score as primary metric (higher is better)
        # For k=1, silhouette is undefined, so use a default score
        score = avg_sil if k > 1 else 0.0
        
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k, results


def cluster_pattern_kmeans(texts, test_ids, k, seed=42):
    """Cluster a pattern using k-means with specified k."""
    if len(texts) < 2 or k < 1:
        return [0] * len(texts), None
    
    try:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf_matrix = vectorizer.fit_transform(texts)
    except:
        return [0] * len(texts), None
    
    k = min(k, len(texts))
    
    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
    else:
        cluster_labels = [0] * len(texts)
    
    return cluster_labels, tfidf_matrix


def analyze_pattern_with_kmeans(pattern, findings, labels):
    """
    Analyze a pattern: find optimal k (unsupervised), cluster, then measure propagation accuracy.
    """
    # Extract texts and labels
    texts = []
    test_ids = []
    true_labels_dict = {}
    
    for finding in findings:
        test_id = finding['test_id']
        if test_id in labels:
            label_data = labels[test_id]
            root_cause = label_data.get('root_cause', 'UNKNOWN')
            texts.append(finding['text'])
            test_ids.append(test_id)
            true_labels_dict[test_id] = root_cause
    
    if len(texts) < 2:
        baseline_acc = 1.0 if len(texts) == 1 else 0.0
        return {
            'optimal_k': 1,
            'propagation_accuracy': baseline_acc,
            'accuracy_std': 0.0,
            'baseline_accuracy': baseline_acc,
            'improvement': 0.0,
            'num_findings': len(texts),
            'unique_root_causes': len(set(true_labels_dict.values())),
            'k_results': {}
        }
    
    # Find optimal k using elbow method (UNSUPERVISED - no labels used)
    optimal_k, k_results = find_optimal_k_elbow(texts)
    
    # Calculate baseline accuracy (all in one cluster)
    baseline_cluster_labels = [0] * len(test_ids)
    try:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        baseline_tfidf = vectorizer.fit_transform(texts)
        baseline_acc, _ = calculate_propagation_accuracy(
            baseline_cluster_labels, test_ids, true_labels_dict, baseline_tfidf
        )
    except:
        baseline_acc = 0.0
    
    # Cluster with optimal k and measure propagation accuracy
    final_accuracies = []
    for seed in range(N_REPEATS):
        cluster_labels, tfidf_matrix = cluster_pattern_kmeans(texts, test_ids, optimal_k, seed)
        if tfidf_matrix is None:
            continue
        
        accuracy, _ = calculate_propagation_accuracy(
            cluster_labels, test_ids, true_labels_dict, tfidf_matrix
        )
        final_accuracies.append(accuracy)
    
    avg_accuracy = np.mean(final_accuracies) if final_accuracies else 0.0
    std_accuracy = np.std(final_accuracies) if len(final_accuracies) > 1 else 0.0
    
    return {
        'optimal_k': optimal_k,
        'propagation_accuracy': avg_accuracy,
        'accuracy_std': std_accuracy,
        'baseline_accuracy': baseline_acc,
        'improvement': avg_accuracy - baseline_acc,
        'num_findings': len(texts),
        'unique_root_causes': len(set(true_labels_dict.values())),
        'k_results': k_results
    }


def main():
    print("=" * 80)
    print("ROOT CAUSE PROPAGATION ACCURACY ANALYSIS")
    print("=" * 80)
    print("Analyzing root cause separation using medoid-based propagation accuracy")
    print("Clustering is performed WITHOUT labels (unsupervised)")
    print()
    
    # Load data
    print("Loading data...")
    labels = load_labels()
    features = load_features()
    
    print(f"Loaded {len(labels)} labels with root causes")
    print(f"Loaded {len(features)} feature vectors")
    
    # Calculate baseline propagation accuracy (exit code patterns)
    print("\n" + "=" * 80)
    print("BASELINE: Exit Code Pattern Level (Medoid Propagation)")
    print("=" * 80)
    print("Calculating baseline propagation accuracy...")
    baseline_accuracies = calculate_baseline_propagation_accuracy(labels)
    
    print(f"\nPattern-level propagation accuracy (baseline):")
    for pattern in sorted(baseline_accuracies.keys()):
        data = baseline_accuracies[pattern]
        print(f"  {pattern}: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%) - "
              f"{data['num_findings']} findings, {data['unique_root_causes']} unique root causes")
    
    # Group findings by pattern
    print("\n" + "=" * 80)
    print("K-MEANS CLUSTERING ANALYSIS (Unsupervised)")
    print("=" * 80)
    print("Finding optimal k using elbow method (no labels used)...")
    
    findings_by_pattern = defaultdict(list)
    for test_id, label_data in labels.items():
        pattern = label_data.get('pattern', 'unknown')
        if test_id in features:
            feat = features[test_id]
            findings_by_pattern[pattern].append({
                'test_id': test_id,
                'text': feat.get('text', '')
            })
    
    # Analyze each pattern with k-means
    kmeans_results = {}
    
    for pattern in sorted(findings_by_pattern.keys()):
        findings = findings_by_pattern[pattern]
        print(f"\nAnalyzing {pattern} ({len(findings)} findings)...")
        
        result = analyze_pattern_with_kmeans(pattern, findings, labels)
        kmeans_results[pattern] = result
        
        print(f"  Optimal k (elbow method): {result['optimal_k']}")
        print(f"  Baseline accuracy: {result['baseline_accuracy']:.4f} ({result['baseline_accuracy']*100:.2f}%)")
        print(f"  K-means accuracy: {result['propagation_accuracy']:.4f} ± {result['accuracy_std']:.4f} ({result['propagation_accuracy']*100:.2f}%)")
        print(f"  Improvement: {result['improvement']:.4f} ({result['improvement']*100:.2f} pp)")
        print(f"  Unique root causes: {result['unique_root_causes']}")
    
    # Calculate overall statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline_avg = np.mean([d['accuracy'] for d in baseline_accuracies.values()])
    kmeans_avg = np.mean([r['propagation_accuracy'] for r in kmeans_results.values()])
    improvement_avg = kmeans_avg - baseline_avg
    
    print(f"\nAverage Propagation Accuracy:")
    print(f"  Baseline (exit code patterns): {baseline_avg:.4f} ({baseline_avg*100:.2f}%)")
    print(f"  With K-means clustering: {kmeans_avg:.4f} ({kmeans_avg*100:.2f}%)")
    print(f"  Average improvement: {improvement_avg:.4f} ({improvement_avg*100:.2f} percentage points)")
    
    # Save results
    results = {
        'baseline_accuracies': baseline_accuracies,
        'kmeans_results': kmeans_results,
        'summary': {
            'baseline_avg_accuracy': float(baseline_avg),
            'kmeans_avg_accuracy': float(kmeans_avg),
            'avg_improvement': float(improvement_avg),
            'num_patterns': len(baseline_accuracies)
        }
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    # Generate report
    with open(REPORT_FILE, 'w') as f:
        f.write("# Root Cause Propagation Accuracy Analysis\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total patterns analyzed: {len(baseline_accuracies)}\n")
        f.write(f"- Average baseline accuracy: {baseline_avg:.4f} ({baseline_avg*100:.2f}%)\n")
        f.write(f"- Average k-means accuracy: {kmeans_avg:.4f} ({kmeans_avg*100:.2f}%)\n")
        f.write(f"- Average improvement: {improvement_avg:.4f} ({improvement_avg*100:.2f} pp)\n\n")
        f.write("## Methodology\n\n")
        f.write("1. **Baseline**: Each exit code pattern is treated as one cluster. ")
        f.write("Medoid's root cause label is propagated to all findings in the pattern.\n")
        f.write("2. **K-means**: Optimal k is found using elbow method (silhouette score) ")
        f.write("**without using labels** (unsupervised). Then k-means clusters the pattern, ")
        f.write("and medoid's root cause label is propagated to all findings in each cluster.\n\n")
        f.write("## Pattern-by-Pattern Analysis\n\n")
        f.write("| Pattern | Baseline Accuracy | K-means Accuracy | Optimal k | Improvement |\n")
        f.write("|---------|------------------|------------------|-----------|-------------|\n")
        
        for pattern in sorted(baseline_accuracies.keys()):
            baseline = baseline_accuracies[pattern]['accuracy']
            kmeans = kmeans_results.get(pattern, {})
            kmeans_accuracy = kmeans.get('propagation_accuracy', 0.0)
            optimal_k = kmeans.get('optimal_k', 1)
            improvement = kmeans.get('improvement', 0.0)
            
            f.write(f"| {pattern} | {baseline:.4f} ({baseline*100:.2f}%) | "
                   f"{kmeans_accuracy:.4f} ({kmeans_accuracy*100:.2f}%) | {optimal_k} | "
                   f"{improvement:.4f} ({improvement*100:.2f} pp) |\n")
    
    print(f"Report saved to {REPORT_FILE}")
    
    # Create visualization
    create_comparison_plot(baseline_accuracies, kmeans_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def create_comparison_plot(baseline_accuracies, kmeans_results):
    """Create visualization comparing baseline vs k-means propagation accuracy."""
    patterns = sorted(baseline_accuracies.keys())
    baseline_values = [baseline_accuracies[p]['accuracy'] for p in patterns]
    kmeans_values = [kmeans_results.get(p, {}).get('propagation_accuracy', 0.0) for p in patterns]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Exit Code Patterns)', 
                   color='#ff9800', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, kmeans_values, width, label='K-means Clustering', 
                   color='#2196f3', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Exit Code Pattern', fontsize=12, fontweight='bold')
    ax.set_ylabel('Propagation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Root Cause Propagation Accuracy: Exit Code Patterns vs K-means Clustering\n(Medoid-based Propagation)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('pattern_', '') for p in patterns], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()

