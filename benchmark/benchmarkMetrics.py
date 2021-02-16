"""
-------------------------------------------------------------------------------
Created: 12.02.2021, 11:34
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Implementation of the relevant metrics to evaluate a given physiological signal, i.e. comparison to some reference.
-------------------------------------------------------------------------------
"""
import numpy as np
from scipy import stats


class BenchmarkMetrics:
    """Combination of all metrics computed within the initialization.
    """

    def __init__(self, extracted, reference):
        """All defined metrics within this class are computed when initialized.

        Parameters
        ----------
        extracted : 1D array like object
            the features extracted from some physiological signal, e.g. the heart rate from iPPG
        reference : 1D array like object
            the features extracted from some reference physiological signal, e.g. the heart rate from continuous blood pressure
        """
        self.extracted = np.squeeze(extracted)
        self.reference = np.squeeze(reference)
        self.metrics = {}
        self.meta = {}

        self.valid = self._get_valid_mask()

        self._calc_mae()
        self._calc_rmse()
        self._calc_pearsonr()
        self._calc_acc()

    def get_metrics(self):
        """Returns all the metrics defined within this class.

        Returns
        -------
        dict
            each entry represents one metric
        """
        return self.metrics

    def get_meta(self):
        """Returns a dict containing information about the number of valid values used for the computation of the metrics.

        Returns
        -------
        dict
            {'num_valid_idxs_extracted', 'num_valid_idxs_reference'} where each value is an integer
        """
        return self.meta

    def _calc_mae(self):
        """Mean of the sum of absolute errors.
        """
        mae = np.mean(
            np.abs(self.extracted[self.valid] - self.reference[self.valid]))
        self.metrics['mae'] = mae

    def _calc_rmse(self):
        """Root of the mean of the sum of squared errors.
        """
        rmse = np.sqrt(
            np.mean((self.extracted[self.valid] - self.reference[self.valid])**2))
        self.metrics['rmse'] = rmse

    def _calc_pearsonr(self):
        """Pearson correlation coefficient.
        """
        r, _ = stats.pearsonr(
            self.extracted[self.valid], self.reference[self.valid])
        self.metrics['pearsonr'] = r

    def _calc_acc(self):
        """Accuracy according to IEC 60601-2-27.

        Following IEC 60601-2-27 (originally for ECG heart rates), 
        a pulse rate was deemed erroneous if the absolute difference 
        between the remotely acquired pulse rate and ground truth pulse 
        rate exceeds the greater of either 5BPM or 10% of the
        ground truth pulse rate.
        """
        abs_diff = np.abs(
            self.extracted[self.valid] - self.reference[self.valid])
        crit = np.zeros((len(abs_diff), 2))
        # 10% of ground truth pulse rate
        crit[:, 0] = 0.1 * self.reference[self.valid]
        crit[:, 1] = 5  # 5bpm
        # exceed the greater of either 5bpm or 10% of ground truth pulse rate
        correct = abs_diff <= np.max(crit, axis=1)
        acc_iec = correct.sum() / correct.size
        self.metrics['acc_iec'] = acc_iec

    def _calc_acc_AUC(self):
        """Area under curve for error (in bpm [1,...,10]) vs. corresponding accuracy as defined in:

        Wang, Wenjin; Den Brinker, Albertus C.; Stuijk, Sander; Haan, Gerard de (2017): Color-Distortion Filtering for Remote Photoplethysmography. In: 12th IEEE International Conference on Automatic Face and Gesture Recognition - FG 2017. pp. 71–78. DOI: 10.1109/FG.2017.18.
        """
        pass

    def _get_valid_mask(self):
        """Build mask to extract only the valid values (not np.nan or np.inf)
        """
        self.invalid_idxs_extracted = np.isnan(
            self.extracted) | np.isinf(self.extracted)
        self.invalid_idxs_reference = np.isnan(
            self.reference) | np.isinf(self.reference)

        self.meta['num_valid_idxs_extracted'] = np.logical_not(
            self.invalid_idxs_extracted).sum()
        self.meta['num_valid_idxs_reference'] = np.logical_not(
            self.invalid_idxs_reference).sum()
        self.meta['num_idxs_overall'] = len(self.extracted)

        return np.logical_not(
            self.invalid_idxs_extracted | self.invalid_idxs_reference)
