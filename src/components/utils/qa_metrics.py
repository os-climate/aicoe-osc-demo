"""QA metrics."""


import numpy as np
from farm.evaluation.metrics import squad_EM, squad_f1
from sklearn.metrics import confusion_matrix


def relaxed_squad_f1(preds, labels):
    """Relax squad f1."""
    # if there is any overlap between prediction and labels, the f1 is considered as as 1.
    f1_scores = []
    n_docs = len(preds)
    for i in range(n_docs):
        best_pred = preds[i][0]
        best_f1 = max(
            [relaxed_squad_f1_single(best_pred, label) for label in labels[i]]
        )
        f1_scores.append(best_f1)
    return np.mean(f1_scores)


def relaxed_squad_f1_single(pred, label, pred_idx=0):
    """Relax squad f1 single."""
    label_start, label_end = label
    span = pred[pred_idx]
    pred_start = span.offset_answer_start
    pred_end = span.offset_answer_end

    if (pred_start + pred_end == 0) or (label_start + label_end == 0):
        if pred_start == label_start:
            return 1.0
        else:
            return 0.0
    pred_span = list(range(pred_start, pred_end + 1))

    label_span = list(range(label_start, label_end + 1))
    n_overlap = len([x for x in pred_span if x in label_span])
    if n_overlap == 0:
        return 0.0
    else:
        return 1.0


def compute_extra_metrics(eval_results):
    """Compute extra metrics."""
    metric_dict = {}
    head_num = 0
    preds = eval_results[head_num]["preds"]
    labels = eval_results[head_num]["labels"]
    is_preds_answerable = [pred_doc[0][0].answer_type == "span" for pred_doc in preds]
    is_labels_answerable = [label_doc != [(-1, -1)] for label_doc in labels]
    # tn: label : unanswerable, predcited: unanswerable
    # fp: label : unanswerable, predcited: answerable
    # fn: label : answerable, predcited: unanswerable
    # fp: label : answerable, predcited: answerable

    tn, fp, fn, tp = confusion_matrix(is_labels_answerable, is_preds_answerable).ravel()
    metric_dict.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

    prediction_answerable_examples = [
        p for doc_idx, p in enumerate(preds) if is_labels_answerable[doc_idx]
    ]
    label_answerable_examples = [
        l for doc_idx, l in enumerate(labels) if is_labels_answerable[doc_idx]
    ]

    relaxed_f1_answerable = relaxed_squad_f1(
        prediction_answerable_examples, label_answerable_examples
    )
    em_answerable = squad_EM(prediction_answerable_examples, label_answerable_examples)
    f1_answerable = squad_f1(prediction_answerable_examples, label_answerable_examples)

    metric_dict.update(
        {
            "relaxed_f1_answerable": relaxed_f1_answerable,
            "em_answerable": em_answerable,
            "f1_answerable": f1_answerable,
        }
    )

    return metric_dict
