"""Independent, role-blind ranking metrics using only the Python standard library."""
import argparse
import json
import math
from pathlib import Path


def ranking_metrics(labels, scores):
    """Threshold-grouped AP, stable Hits@L, and tie-aware Mann–Whitney AUC."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('controller split must contain positive and negative candidates')
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    true_positives = 0
    ap = 0.0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group_positive = sum(labels[i] for i in order[start:end])
        group_negative = end - start - group_positive
        # Each negative ranks below all earlier positives and ties this group's positives.
        concordant += group_negative * (true_positives + .5 * group_positive)
        true_positives += group_positive
        ap += (group_positive / positives) * (true_positives / end)
        start = end
    return {
        'average_precision': ap,
        'hits_at_l': sum(labels[i] for i in order[:positives]) / positives,
        'roc_auc': concordant / (positives * negatives),
    }


def score(instance, solution):
    schema_error = int(not isinstance(solution, dict) or
                       solution.get('schema_version') != 'local-link-prediction-solution.v1')
    raw = solution.get('scores') if isinstance(solution, dict) else None
    if not isinstance(raw, list):
        schema_error += 1
        raw = []
    non_finite = 0
    for item in raw:
        try:
            valid = type(item) in (int, float) and math.isfinite(float(item))
        except OverflowError:
            valid = False
        non_finite += int(not valid)
    mismatch = abs(len(raw) - len(instance['candidates']))
    metrics = {'schema_error_count': schema_error, 'score_count_mismatch': mismatch,
               'non_finite_score_count': non_finite,
               'average_precision': -1.0, 'hits_at_l': -1.0, 'roc_auc': -1.0}
    if not (schema_error or mismatch or non_finite):
        hidden = {tuple(edge) for edge in instance['hidden_edges']}
        labels = [int(tuple(edge) in hidden) for edge in instance['candidates']]
        metrics.update(ranking_metrics(labels, raw))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    args = parser.parse_args()
    try:
        solution = json.loads(args.solution.read_text())
    except (ValueError, OSError):
        solution = None
    print(json.dumps({'metrics': score(json.loads(args.instance.read_text()), solution)},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
