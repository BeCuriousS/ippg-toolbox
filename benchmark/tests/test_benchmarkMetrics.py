"""
-------------------------------------------------------------------------------
Created: 12.02.2021, 12:39
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Testing the metrics
-------------------------------------------------------------------------------
"""
# %%
import numpy as np
from benchmark import BenchmarkMetrics


class TestBenchmarkMetrics:

    def __init__(self):
        self.extracted = np.arange(0, 20)
        self.reference = np.arange(0, 20) + 6
        self.bM = BenchmarkMetrics(self.extracted, self.reference)
        print(self.bM.get_metrics())
        print(self.bM.get_meta())


if __name__ == '__main__':

    testBenchmarkMetrics = TestBenchmarkMetrics()

# %%
