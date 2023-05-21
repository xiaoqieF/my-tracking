import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm

mm.lap.default_solver = 'lap'
metrics = list(mm.metrics.motchallenge_metrics)

gt_file = './run/2_gt.txt'
track_file = './run/2_bot.txt'

mot_type = 'bot'

gt_files = ['./run/1_gt.txt','./run/2_gt.txt', './run/3_gt.txt', './run/4_gt.txt', './run/5_gt.txt']
track_files = [g.replace('gt', mot_type) for g in gt_files]

accs = []
names = []
for g, t in zip(gt_files, track_files):
    gt = mm.io.loadtxt(g, fmt='mot16', min_confidence=-1)
    ts = mm.io.loadtxt(t, fmt='mot16')
    
    #计算单个acc
    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    accs.append(acc)
    name=os.path.splitext(os.path.basename(t))[0]
    names.append(name)
mh = mm.metrics.create()
summary = mh.compute_many(accs, metrics=metrics, names=names)
print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))