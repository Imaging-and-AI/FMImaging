"""
Support functions for generic metric manager
"""

import torchmetrics
import numpy as np
import warnings

# -------------------------------------------------------------------------------------------------
def get_metric_function(metric_name, num_classes, metric_task, multidim_average):
    """
    Returns the function to compute the metric specified by metric_name
    """

    if metric_name=='acc_1':
        return torchmetrics.Accuracy(task=metric_task, num_classes=num_classes, top_k=1, average='micro', multidim_average=multidim_average)
    elif metric_name=='auroc':
        return torchmetrics.AUROC(task=metric_task, num_classes=num_classes, average='macro')
    elif metric_name=='f1':
         return torchmetrics.F1Score(task=metric_task, num_classes=num_classes, average='macro', multidim_average=multidim_average) # Will not be exact for multiclass dice, but will be close
    elif metric_name=='psnr':
        warnings.warn("WARNING: Data range not specified in default PSNR metric; may impact results (particularly with batch size) if data is not normalized")
        return torchmetrics.image.PeakSignalNoiseRatio()
    elif metric_name=='ssim':
        warnings.warn("WARNING: Data range not specified in default SSIM metric; may impact results (particularly with batch size) if data is not normalized")
        return torchmetrics.image.StructuralSimilarityIndexMeasure()
    else:
        raise NotImplementedError('Unknown metric type specified:',metric_name)
    

# -------------------------------------------------------------------------------------------------
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.counts = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.vals.append(val)
        self.counts.append(n)

    def status(self):
        return np.array(self.vals), np.array(self.counts)


