#from https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/classification/calibration_error.py

"""
    Copyright 2020-2022 Lightning-AI team

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import torch
def _binning_bucketize(confidences, accuracies, bin_boundaries):
    """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
    count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

    indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()
    return acc_bin, conf_bin, prop_bin