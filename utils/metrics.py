import numpy as np

def calculate_ece(y_true, y_prob, n_bins=15):
    """Calculates Expected Calibration Error in a robust way."""
    if len(y_true) == 0:
        return 0.0
    
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = binids == i
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            avg_accuracy_in_bin = np.mean(y_true[in_bin])
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
            
    return ece
