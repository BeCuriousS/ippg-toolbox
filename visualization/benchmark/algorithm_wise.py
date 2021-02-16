"""
-------------------------------------------------------------------------------
Created: 15.02.2021, 09:53
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Visualize the benchmark results method wise.
-------------------------------------------------------------------------------
"""
import csv
import pickle
from scipy.io import loadmat, savemat
import numpy as np
import os
from pathlib import Path
import traceback
import progressbar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from benchmark import settings


class Plotter:

    def __init__(self, df_dir, prefix, metrics_labels):
        self.df = pd.read_pickle(df_dir).transpose()
        self.plot_handles = []
        self.summary = []
        self.save_fig_path = 'reports/figures/benchmark/algorithms'
        self.save_tbl_path = 'reports/tables/benchmark/algorithms'
        Path(self.save_fig_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_tbl_path).mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.metrics_labels = metrics_labels
        # was color distortion used for the benchmark?
        self.suffix = ''
        if settings.APPLY_COLOR_DISTORTION_FILTERING():
            self.suffix += '_cdf'
        # process
        self._prepare_dataframe()
        self._create_plots()
        self._create_tables()

    def _prepare_dataframe(self):
        cols = self.df.columns
        for c in cols:
            if 'meta' in c:
                self.df = self.df.drop(c, axis=1)

    def _create_plots(self):
        fig_sp, axs_sp = plt.subplots(1, len(self.metrics_labels), figsize=(
            FIGSIZE[0]*len(self.metrics_labels), FIGSIZE[1]), dpi=DPI, tight_layout=True)

        # apply median filter to get one value for snr columns
        df = self.df.applymap(np.median)

        for i, m in enumerate(self.metrics_labels):
            fig, axs = plt.subplots(
                1, 1, figsize=FIGSIZE, dpi=DPI, tight_layout=True)
            cols = []
            for c in self.df.columns:
                if c.endswith('_'+m):
                    cols.append(c)

            sns.boxplot(data=df[cols], ax=axs, saturation=1)
            sns.boxplot(data=df[cols], ax=axs_sp[i], saturation=1)

            if m == 'mae':
                axs.set_ylabel('Beats per minute')
                axs_sp[i].set_ylabel('Beats per minute')
                _ = axs.set_title('Mean absolute error (MAE)')
                _ = axs_sp[i].set_title('Mean absolute error (MAE)')
            elif m == 'rmse':
                axs.set_ylabel('Beats per minute')
                axs_sp[i].set_ylabel('Beats per minute')
                _ = axs.set_title('Root of the mean of squared errors (RMSE)')
                _ = axs_sp[i].set_title(
                    'Root of the mean of squared errors (RMSE)')
            elif m == 'snr':
                axs.set_ylabel('dB')
                axs_sp[i].set_ylabel('dB')
                _ = axs.set_title('Signal to noise ratio (SNR)')
                _ = axs_sp[i].set_title('Signal to noise ratio (SNR)')
            elif m == 'snr_vs_ref':
                axs.set_ylabel('dB')
                axs_sp[i].set_ylabel('dB')
                _ = axs.set_title('Signal to noise ratio (SNR) vs. reference')
                _ = axs_sp[i].set_title(
                    'Signal to noise ratio (SNR) vs. reference')
            elif m == 'pearsonr':
                axs.set_ylabel('No unit')
                axs_sp[i].set_ylabel('No unit')
                _ = axs.set_title('Pearson correlation coefficient (r)')
                _ = axs_sp[i].set_title('Pearson correlation coefficient (r)')
            elif m == 'acc_iec':
                axs.set_ylabel('No unit')
                axs_sp[i].set_ylabel('No unit')
                _ = axs.set_title('Accuracy (ACC)')
                _ = axs_sp[i].set_title('Accuracy (ACC)')
            axs.set_xticklabels([c.split('_')[0] for c in cols])
            axs_sp[i].set_xticklabels([c.split('_')[0] for c in cols])

            axs.grid(True)
            axs_sp[i].grid(True)
            self.plot_handles.append([axs, fig, m])
        self.plot_handles.append([axs_sp, fig_sp])
        self._save_figures()

    def _create_tables(self):
        # apply median filter to get one value for snr columns
        df = self.df.applymap(np.median)
        cols = []
        for i, m in enumerate(self.metrics_labels):
            for c in df.columns:
                if c.endswith('_'+m):
                    cols.append(c)
        # print summary to std out and file
        full_file_path = os.path.join(
            self.save_tbl_path, self.prefix+'_summary' + self.suffix + '.csv')
        df_tbl = [df[cols].mean().round(decimals=2),
                  df[cols].std().round(decimals=2),
                  df[cols].median().round(decimals=2)]
        df_tbl = pd.concat(df_tbl, axis=1)
        df_tbl.columns = ['mean', 'std', 'median']
        df_tbl.to_csv(full_file_path)

    def _save_figures(self):
        for ph in self.plot_handles[:-1]:
            plt.figure(ph[1].number)
            plt.savefig(os.path.join(self.save_fig_path,
                                     self.prefix + '_' + ph[2] + self.suffix + '.svg'))
        plt.figure(self.plot_handles[-1][1].number)
        plt.savefig(os.path.join(self.save_fig_path,
                                 self.prefix + '_' + 'allInOne' + self.suffix + '.svg'))


if __name__ == '__main__':
    # ----------------------------------------------------------
    # plot settings
    # ----------------------------------------------------------
    plt.rc('axes', axisbelow=True)
    plt.style.use('seaborn-dark')
    sns.set_context("paper")
    FIGSIZE = (3, 8)
    DPI = 256

    metrics_labels = [
        'mae',
        'rmse',
        'snr',
        'snr_vs_ref',
        'acc_iec',
        'pearsonr'
    ]

    # ----------------------------------------------------------
    # process
    # ----------------------------------------------------------
    df_dir_ubfc = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/evaluation/eval_dataframe.p'
    df_dir_bp4d = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/evaluation/eval_dataframe.p'

    Plotter(df_dir_ubfc, 'UBFC', metrics_labels)
    Plotter(df_dir_bp4d, 'BP4D', metrics_labels)
