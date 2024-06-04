import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import torch.nn as nn
import torch
import torch.nn.functional as F

def auroc(preds, labels, pos_label=1):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    return auc(fpr, tpr)


def aupr(preds, labels, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels, pos_label=1):
    """Return the misclassification probability when TPR is 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    # Get ratios of positives to negatives
    pos_ratio = sum(np.array(labels) == pos_label) / len(labels)
    neg_ratio = 1 - pos_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    # Calc is the (# of negatives * FNR) + (# of positives * FPR)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))

def accuracy_at_tpr95(preds, pos_labels=1):
    
    real_labels = np.load('./ImageNet_labels_50000.npy')
    labels = np.zeros(len(real_labels))
    labels += 1
    
    
    max_preds = np.max(preds, axis=1)
    max_idx = np.argmax(preds, axis=1)
    
    _, tpr, thresholds = roc_curve(labels, max_preds, pos_label=pos_labels)
    
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
    
    idx_tpr95 = (max_preds >= thresholds[idxs][0])
    
    return thresholds[idxs][0], (max_idx[idx_tpr95] == real_labels[idx_tpr95]).sum()/len(real_labels[idx_tpr95])
    

def calc_metrics(predictions, labels, pos_label=1):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.
    
    These metrics conform to how results are reported in the paper 'Enhancing The 
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.
    
        preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
        labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

        pos_label: label of the positive class (1 by default)
    """

    return {
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels, pos_label=pos_label),
        'detection_error': detection_error(predictions, labels, pos_label=pos_label),
        'auroc': auroc(predictions, labels, pos_label=pos_label),
        'aupr_in': aupr(predictions, labels, pos_label=pos_label),
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels], pos_label=pos_label),
    }
    
def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array([y for i in _pos for y in i]).reshape((-1,1))
    neg = np.array([y for i in _neg for y in i]).reshape((-1,1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    ood = calc_metrics(examples, labels)
    return ood


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        #### original
        # softmaxes = F.softmax(logits, dim=1)
        # confidences, predictions = torch.max(softmaxes, 1)
        ####
        confidences, predictions = torch.max(logits, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
