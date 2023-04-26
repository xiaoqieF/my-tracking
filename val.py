import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm

mm.lap.default_solver = 'lap'
metrics = list(mm.metrics.motchallenge_metrics)

gt_file = './run/2_gt.txt'
track_file = './run/2_bot.txt'

gt = mm.io.loadtxt(gt_file, fmt='mot16', min_confidence=-1)
ts = mm.io.loadtxt(track_file, fmt='mot16')
name=os.path.splitext(os.path.basename(track_file))[0]
#计算单个acc
acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=metrics, name=name)
print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))