from evaluate import load
from collections import Counter

bert_score_metric = load("bertscore")

def f1_score(prediction: str, reference: str):
    # Compute character-level F1 score
    # Count occurrences for each character
    pred_counter = Counter(prediction)
    ref_counter = Counter(reference)
    
    # Calculate the number of overlapping characters
    common = sum(min(pred_counter[char], ref_counter[char]) for char in pred_counter)
    
    # If there is no overlap, return 0.0 immediately
    if common == 0:
        return 0.0
    
    precision = common / len(prediction) if prediction else 0.0
    recall = common / len(reference) if reference else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    # Return the F1 score
    return 2 * precision * recall / (precision + recall)


def bert_score(prediction: str, reference: str):
    return bert_score_metric.compute(predictions=[prediction], references=[reference], lang="en")["f1"][0]