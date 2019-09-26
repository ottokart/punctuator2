# coding: utf-8
"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

from __future__ import print_function

from numpy import nan
import data
import sys
from io import open

# Can be used to estimate 2-class performance for example
MAPPING = {
    # "!EXCLAMATIONMARK": ".PERIOD",
    # "?QUESTIONMARK": ".PERIOD",
    # ":COLON": ".PERIOD",
    # ";SEMICOLON": ".PERIOD"
}


def compute_error(target_paths, predicted_paths):
    counter = 0
    total_correct = 0

    correct = 0.
    substitutions = 0.
    deletions = 0.
    insertions = 0.

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = " "
        predicted_punctuation = " "

        t_i = 0
        p_i = 0

        with open(target_path, 'r', encoding='utf-8') as target, open(predicted_path, 'r', encoding='utf-8') as predicted:

            target_stream = target.read().split()
            predicted_stream = predicted.read().split()

            while True:

                if data.PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in data.PUNCTUATION_VOCABULARY:
                    while data.PUNCTUATION_MAPPING.get(
                        target_stream[t_i], target_stream[t_i]
                    ) in data.PUNCTUATION_VOCABULARY: # skip multiple consecutive punctuations
                        target_punctuation = data.PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                        target_punctuation = MAPPING.get(target_punctuation, target_punctuation)
                        t_i += 1
                else:
                    target_punctuation = " "

                if predicted_stream[p_i] in data.PUNCTUATION_VOCABULARY:
                    predicted_punctuation = MAPPING.get(predicted_stream[p_i], predicted_stream[p_i])
                    p_i += 1
                else:
                    predicted_punctuation = " "

                is_correct = target_punctuation == predicted_punctuation

                counter += 1
                total_correct += is_correct

                if predicted_punctuation == " " and target_punctuation != " ":
                    deletions += 1
                elif predicted_punctuation != " " and target_punctuation == " ":
                    insertions += 1
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation == target_punctuation:
                    correct += 1
                elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation != target_punctuation:
                    substitutions += 1

                true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(is_correct)
                false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(not is_correct)
                false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(not is_correct)

                assert target_stream[t_i] == predicted_stream[p_i] or predicted_stream[p_i] == "<unk>", \
                       ("File: %s \n" + \
                       "Error: %s (%s) != %s (%s) \n" + \
                       "Target context: %s \n" + \
                       "Predicted context: %s") % \
                       (target_path,
                        target_stream[t_i], t_i, predicted_stream[p_i], p_i,
                        " ".join(target_stream[t_i-2:t_i+2]),
                        " ".join(predicted_stream[p_i-2:p_i+2]))

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream) - 1 and p_i >= len(predicted_stream) - 1:
                    break

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-" * 46)
    print("{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION', 'PRECISION', 'RECALL', 'F-SCORE'))
    for p in data.PUNCTUATION_VOCABULARY:

        if p == data.SPACE:
            continue

        overall_tp += true_positives.get(p, 0.)
        overall_fp += false_positives.get(p, 0.)
        overall_fn += false_negatives.get(p, 0.)

        punctuation = p
        precision = (true_positives.get(p, 0.) / (true_positives.get(p, 0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p, 0.) / (true_positives.get(p, 0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan
        print(u"{:<16} {:<9} {:<9} {:<9}".format(punctuation, round(precision, 3) * 100, round(recall, 3) * 100, round(f_score, 3) * 100).encode('utf-8'))
    print("-" * 46)
    pre = overall_tp / (overall_tp + overall_fp) if overall_fp else nan
    rec = overall_tp / (overall_tp + overall_fn) if overall_fn else nan
    f1 = (2. * pre * rec) / (pre + rec) if (pre + rec) else nan
    print("{:<16} {:<9} {:<9} {:<9}".format("Overall", round(pre, 3) * 100, round(rec, 3) * 100, round(f1, 3) * 100))
    print("Err: %s%%" % round((100.0 - float(total_correct) / float(counter - 1) * 100.0), 2))
    print("SER: %s%%" % round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        sys.exit("Ground truth file path argument missing")

    if len(sys.argv) > 2:
        predicted_path = sys.argv[2]
    else:
        sys.exit("Model predictions file path argument missing")

    compute_error([target_path], [predicted_path])
