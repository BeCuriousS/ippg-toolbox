"""
 -----------------------------------------------------------------------------
Created: 10.02.2021, 18:47
 -----------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
 -----------------------------------------------------------------------------
Purpose: Execute to run the full benchmark and create a pandas dataframe to
ease further plotting steps.
 -----------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import progressbar
from pathlib import Path
from scipy.io import loadmat, savemat
import warnings
import traceback
import pandas as pd

from ippgtoolbox.benchmark import BenchmarkAlgorithms
from ippgtoolbox.benchmark import BenchmarkMetrics
from ippgtoolbox.benchmark import ProcessExtraction
from ippgtoolbox.benchmark import settings
from ippgtoolbox import processing


class ProcessUBFC:
    """Combination of methods to run the full benchmark on the UBFC database.
    """

    def __init__(self, dst_dir):
        # ----------------------------------------------------------
        # vars
        # ----------------------------------------------------------
        self.df = {}
        self.dst_dir = dst_dir
        # ----------------------------------------------------------
        # shortcuts
        # ----------------------------------------------------------
        self.use_cdf = settings.APPLY_COLOR_DISTORTION_FILTERING()
        # ----------------------------------------------------------
        # pipeline
        # ----------------------------------------------------------
        self._set_vars()
        self._collect_record_names()
        self._process_records()
        self._save_to_dataframe()

    def _set_vars(self):
        # ----------------------------------------------------------
        # set paths
        # ----------------------------------------------------------
        self.db_dir = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/measurements'
        self.mean_rgb_dir = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/MEAN_RGB_deepLab'
        # ----------------------------------------------------------
        # set db specific vars
        # ----------------------------------------------------------
        self.sample_freq = 30
        self.ref_is_hr_seq = False

    def _collect_record_names(self):
        self.record_names = [rn for rn in os.listdir(self.db_dir)
                             if 'subject' in rn]

    def _process_records(self):
        bar = progressbar.ProgressBar(
            max_value=len(self.record_names)-1,
            variables={'subj_name': ''},
            suffix=' >>> Processing record {variables.subj_name:12s}',
            term_width=120)

        for i, rn in enumerate(self.record_names):
            bar.update(i, subj_name=rn)
            self.record_name = rn
            self.full_dst_dir = os.path.join(self.dst_dir, rn)
            Path(self.full_dst_dir).mkdir(parents=True, exist_ok=True)
            try:
                self._read_reference()
                self._read_extracted()
                self._compute_benchmark_algorithms()
                self._process_extraction()
                self._compute_benchmark_metrics()
                self._add_to_dataframe()
            except:
                warnings.warn('Could not process this record', UserWarning)
                traceback.format_exc()

    def _save_to_dataframe(self):
        self.df = pd.DataFrame(self.df)
        self.df.to_pickle(os.path.join(self.dst_dir, 'eval_dataframe.p'))

    def _read_reference(self):
        # resample with timestamps to compensate possible unsteady fps (each recorded frame corresponds to one ppg sample point and timestamp)
        ref_data = np.loadtxt(os.path.join(
            self.db_dir, self.record_name, 'ground_truth.txt'))
        ref_seq = ref_data[0, :]
        ref_ts = ref_data[2, :] * 1e6  # to micros
        tmp = processing.resample_sequence(
            ref_seq, self.sample_freq, seq_ts=ref_ts)
        self.ref_ts = tmp['ts_interp']
        self.ref_seq = tmp['seq_interp']

    def _read_extracted(self):
        data = loadmat(os.path.join(
            self.mean_rgb_dir, self.record_name, 'mean_rgb.mat'))
        mean_rgb_ts = data['timestamps']
        mean_rgb_seq = data['mean_rgb']
        # resample with timestamps to compensate possible unsteady fps (each recorded frame corresponds to one ppg sample point and timestamp)
        mean_rgb_seq_interp = []
        for i in range(mean_rgb_seq.shape[1]):
            tmp = processing.resample_sequence(
                mean_rgb_seq[:, i], self.sample_freq, seq_ts=mean_rgb_ts)
            mean_rgb_seq_interp.append(tmp['seq_interp'])
        self.mean_rgb_ts = tmp['ts_interp']
        self.mean_rgb_seq = np.transpose(np.asarray(mean_rgb_seq_interp))

    def _compute_benchmark_algorithms(self):
        bA = BenchmarkAlgorithms(
            self.mean_rgb_seq,
            self.mean_rgb_ts,
            self.sample_freq,
            apply_cdf=self.use_cdf)
        bvp_data = bA.extract_all()
        # save
        for key, val in bvp_data.items():
            savemat(os.path.join(
                self.full_dst_dir, 'bvp_{}.mat'.format(key)), val)

    def _process_extraction(self):
        # iterate over all available sequences (those that have not just been created as well)
        data = {}
        for fn in os.listdir(self.full_dst_dir):
            if fn.startswith('bvp_'):
                tmp = loadmat(os.path.join(
                    self.dst_dir, self.record_name, fn))
                name = fn[fn.find('_')+1:fn.find('.')]
                if 'DeepPerfusion' in name:
                    # bvp[name] = np.squeeze(bvp[name])
                    # bvp[name] = processing.resample_sequence(
                    #     bvp[name], 30, sample_freq=25)
                    continue
                data[name] = {}
                data[name][name] = tmp[name].squeeze()
                data[name]['timestamps'] = tmp['timestamps'].squeeze()
        # adjust to equal number of values (due to windowing there can occur small variations). It is assumed that all signals interpolated to an equal sample frequency
        ts_max = np.inf
        ts_min = -1
        for key, val in data.items():
            if ts_max > val['timestamps'][-1]:
                ts_max = val['timestamps'][-1].copy()
            if ts_min < val['timestamps'][0]:
                ts_min = val['timestamps'][0].copy()
        for key, val in data.items():
            indexes = np.logical_and(
                val['timestamps'] >= ts_min, val['timestamps'] <= ts_max)
            data[key][key] = val[key][indexes]
        indexes = np.logical_and(self.ref_ts >= ts_min, self.ref_ts <= ts_max)
        self.ref_seq = self.ref_seq[indexes]
        # compute
        pE = ProcessExtraction(
            self.ref_seq, self.sample_freq, ref_is_hr_seq=self.ref_is_hr_seq)
        for key, val in data.items():
            pE.compute_features(val[key], self.sample_freq, key)
        self.report = pE.get_report()
        self.report['skin_color'] = np.mean(self.mean_rgb_seq, axis=0)
        # save
        savemat(os.path.join(
            self.full_dst_dir, 'process_report.mat'), self.report)

    def _compute_benchmark_metrics(self):
        reference_hr = self.report['ref_hr']
        self.metrics = {}
        for key, val in self.report.items():
            if key.endswith('_hr') and key != 'ref_hr':
                bM = BenchmarkMetrics(val, reference_hr)
                m = bM.get_metrics()
                for key_, val_ in m.items():
                    self.metrics[key.split('_hr')[0]+'_'+key_] = val_
                self.metrics[key.split('_hr')[0]+'_meta'] = bM.get_meta()
        # save
        savemat(os.path.join(
            self.full_dst_dir, 'metrics.mat'), self.metrics)

    def _add_to_dataframe(self):
        for key, val in self.report.items():
            if key.endswith('_hr') or \
               key.endswith('_snr') or \
               key.endswith('_snr_vs_ref') or \
               key == 'skin_color':
                self.df.setdefault(self.record_name, {})[key] = val
        for key, val in self.metrics.items():
            self.df.setdefault(self.record_name, {})[key] = val


class ProcessBP4D(ProcessUBFC):
    """Combination of methods to run the full benchmark on the BP4D+ database. Use this class, implement the methods you need and set the variables accordingly.

    Parameters
    ----------
    ProcessUBFC : class
        derived for convenience.
    """

    def _set_vars(self):
        # ----------------------------------------------------------
        # set paths
        # ----------------------------------------------------------
        self.db_dir = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/measurements'
        self.mean_rgb_dir = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/MEAN_RGB_deepLab'
        # ----------------------------------------------------------
        # set db specific vars
        # ----------------------------------------------------------
        self.sample_freq = 25
        self.ref_is_hr_seq = True

    def _collect_record_names(self):
        self.record_names = [rn for rn in os.listdir(self.db_dir)
                             if 'F' in rn or 'M' in rn]

    def _read_reference(self):
        # resample to lower sample frequency to speed up computation
        # ref_seq = np.loadtxt(os.path.join(
        #     self.db_dir, self.record_name, 'signals', 'BP_mmHg.txt'))
        ref_seq = np.loadtxt(os.path.join(
            self.db_dir, self.record_name, 'signals', 'Pulse Rate_BPM.txt'))
        tmp = processing.resample_sequence(
            ref_seq, self.sample_freq, sample_freq=1000)
        self.ref_ts = tmp['ts_interp']
        self.ref_seq = tmp['seq_interp']

    def _read_extracted(self):
        data = loadmat(os.path.join(
            self.mean_rgb_dir, self.record_name, 'mean_rgb.mat'))
        self.mean_rgb_ts = data['timestamps']
        self.mean_rgb_seq = data['mean_rgb']


if __name__ == '__main__':

    # df_dir_ubfc = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/evaluation'
    # ProcessUBFC(df_dir_ubfc)

    df_dir_bp4d = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/evaluation'
    ProcessBP4D(df_dir_bp4d)
