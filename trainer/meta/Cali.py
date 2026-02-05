import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# calibrators
class IsotonicCalibrator:
    """Isotonic regression calibrator - non-parametric, monotonic."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, y_prob, y_true):
        self.iso.fit(y_prob, y_true)
        return self
        
    def predict(self, y_prob):
        return self.iso.predict(y_prob)
 
 
class PlattCalibrator:
    """Platt scaling - logistic regression on probabilities."""
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
        
    def fit(self, y_prob, y_true):
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        self.lr.fit(y_prob.reshape(-1, 1), y_true)
        return self
        
    def predict(self, y_prob):
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        return self.lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
 
 
class BetaCalibrator:
    """Beta calibration - more flexible than Platt."""
    def __init__(self):
        self.a, self.b, self.c = 1.0, 1.0, 0.0
        
    def fit(self, y_prob, y_true):
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        
        def neg_ll(params):
            a, b, c = params
            logit_cal = c + a * np.log(y_prob) - b * np.log(1 - y_prob)
            p_cal = np.clip(expit(logit_cal), 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))
        
        from scipy.optimize import minimize
        result = minimize(neg_ll, [1.0, 1.0, 0.0], method='L-BFGS-B',
                         bounds=[(0.01, 10), (0.01, 10), (-5, 5)])
        self.a, self.b, self.c = result.x
        return self
        
    def predict(self, y_prob):
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logit_cal = self.c + self.a * np.log(y_prob) - self.b * np.log(1 - y_prob)
        return expit(logit_cal)
 
 
def get_calibrator(cal_method):
    """Factory function to create calibrator by name."""
    if cal_method == 'isotonic':
        return IsotonicCalibrator()
    elif cal_method == 'platt':
        return PlattCalibrator()
    elif cal_method == 'beta':
        return BetaCalibrator()
    else:
        return None

