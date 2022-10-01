import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def get_scores(model, drug_smiles, val_edges, val_labels):
    model.eval()
    preds = model(drug_smiles, val_edges)

    y_label = val_labels.tolist()
    y_pred = preds.tolist()
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    roc_score = roc_auc_score(y_label, y_pred)
    ap_score = average_precision_score(y_label, y_pred)
    f1_score_ = f1_score(y_label, outputs)
    acc_score = accuracy_score(y_label, outputs)
    return roc_score, ap_score, f1_score_, acc_score

def get_scores_main(model, val_data, val_labels):
    model.eval()
    preds = model(val_data)

    y_label = val_labels.tolist()
    y_pred = preds.tolist()
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    roc_score = roc_auc_score(y_label, y_pred)
    ap_score = average_precision_score(y_label, y_pred)
    f1_score_ = f1_score(y_label, outputs)
    acc_score = accuracy_score(y_label, outputs)
    return roc_score, ap_score, f1_score_, acc_score