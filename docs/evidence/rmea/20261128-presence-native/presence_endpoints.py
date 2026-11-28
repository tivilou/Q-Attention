"""Prespecified secondary task and fixed-Q intervention diagnostics."""
import numpy as np

DIRECTION_GATES = ('fewer_null_false_positives','positive_correct_preserved','positive_top3_preserved')


def task_errors(logits, targets, weights):
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    class_weight = np.zeros(40, dtype=np.float64)
    class_weight[targets] = weights
    if not np.array_equal(class_weight[targets], weights):
        raise ValueError('The frozen assessment requires one weight per class')
    pred = np.asarray(logits).argmax(axis=-1)
    positive = targets != 0
    gold = targets[positive]
    scores = logits[positive,1:]
    gold_score = scores[np.arange(len(gold)),gold-1]
    ids = np.arange(1,40)[None,:]
    outranks = (scores > gold_score[:,None]) | ((scores == gold_score[:,None]) & (ids < gold[:,None]))
    top3 = np.zeros(len(targets), dtype=bool)
    top3[positive] = outranks.sum(axis=1) < 3
    events = {'null_false_positive':~positive & (pred!=0), 'positive_correct':positive & (pred==targets),
              'positive_top3':top3, 'positive_to_null':positive & (pred==0),
              'positive_to_wrong_positive':positive & (pred!=0) & (pred!=targets)}
    return {k:{'count':int(v.sum()), 'weighted_mass':float(class_weight @ np.bincount(targets[v],minlength=40))}
            for k,v in events.items()}


def task_prediction(metrics, candidate, controls):
    q=metrics[candidate]
    return {c:{'fewer_null_false_positives':q['null_false_positive']['weighted_mass'] < metrics[c]['null_false_positive']['weighted_mass'],
               'positive_correct_preserved':q['positive_correct']['weighted_mass'] >= metrics[c]['positive_correct']['weighted_mass'],
               'positive_top3_preserved':q['positive_top3']['weighted_mass'] >= metrics[c]['positive_top3']['weighted_mass']}
            for c in controls}
