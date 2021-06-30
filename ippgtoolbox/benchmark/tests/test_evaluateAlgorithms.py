"""
-------------------------------------------------------------------------------
Created: 12.02.2021, 10:36
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests the implementation of the evaluation for the extracted blood volume pulse vs a given reference.
-------------------------------------------------------------------------------
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ippgtoolbox.benchmark import ProcessExtraction

TEST_sample_freq_ref_bp = 1000
TEST_abs_file_path_ref_bp = os.path.join(
    './assets/bp4d_example/BP_mmHg.txt')
TEST_sample_freq_ref_hr_seq = 1000
TEST_abs_file_path_ref_hr_seq = os.path.join(
    './assets/bp4d_example/Pulse Rate_BPM.txt')
TEST_sample_freq_bvp = 25
TEST_abs_file_path_bvp = os.path.join(
    './assets/bp4d_example/bvp_CHROM_cdf.mat')

plt.rc('lines', linewidth=0.2)


class TestEvaluateAlgorithms:

    def __init__(self):
        self.bp = np.loadtxt(TEST_abs_file_path_ref_bp)
        self.hr_seq = np.loadtxt(TEST_abs_file_path_ref_hr_seq)
        self.bvp = loadmat(TEST_abs_file_path_bvp)['CHROM']
        self._test_plot_seq(self.bp, 'Continuous blood pressure')
        self._test_plot_seq(self.hr_seq, 'Reference HR sequence')
        self._test_plot_seq(self.bvp, 'Extracted blood volume pulse')

    def test_init(self):
        self.eA = ProcessExtraction(
            self.bp,
            ref_sample_freq=TEST_sample_freq_ref_bp,
        )
        self._test_plot_seq(self.eA.ref_hr, 'Reference heart rate from cbp')
        self._test_plot_seq(self.eA.ref_snr, 'Reference snr from cbp')
        self.eA = ProcessExtraction(
            self.bp,
            ref_sample_freq=TEST_sample_freq_ref_bp,
            resample_freq=30,
        )
        self._test_plot_seq(
            self.eA.ref_hr, 'Reference heart rate from cbp down sampled')
        self._test_plot_seq(
            self.eA.ref_snr, 'Reference snr from cbp down sampled')
        self.eA = ProcessExtraction(
            self.hr_seq,
            ref_sample_freq=TEST_sample_freq_ref_hr_seq,
            ref_is_hr_seq=True,
        )
        self._test_plot_seq(
            self.eA.ref_hr, 'Reference heart rate from ref hr seq')

    def test_compute_features_and_metrics(self):
        self.eA = ProcessExtraction(
            self.bp,
            ref_sample_freq=TEST_sample_freq_ref_bp,
            resample_freq=30,
        )
        self.eA.compute_features_and_metrics(
            self.bvp, TEST_sample_freq_bvp, 'TESTPREFIX')
        rp = self.eA.get_report()
        self._test_plot_seq(rp['ref_hr'], 'Reference heart rate')
        self._test_plot_seq(rp['ref_snr'], 'Reference snr')
        self._test_plot_seq(rp['TESTPREFIX_hr'], 'TESTPREFIX heart rate')
        self._test_plot_seq(rp['TESTPREFIX_snr'], 'TESTPREFIX snr')
        self._test_plot_seq(rp['TESTPREFIX_snr_vs_ref'],
                            'TESTPREFIX snr vs ref')

    def _test_plot_seq(self, seq, title):
        plt.figure()
        plt.title(title)
        plt.plot(seq)


if __name__ == '__main__':

    testEvaluateAlgorithms = TestEvaluateAlgorithms()
    testEvaluateAlgorithms.test_init()
    testEvaluateAlgorithms.test_compute_features_and_metrics()

# %%
