#!/usr/bin/env python3

import subprocess
import re


num_threads = "56"

codes = [ \
        'fdbscan-openmp', \
        'fdbscan-dense-openmp', \
        'fdbscan-hip', \
        'fdbscan-dense-hip', \
        'tepp' \
         ]

executables = {}
executables['fdbscan-openmp'] = 'OMP_NUM_THREADS=' + num_threads + ' ./arborx_openmp'
executables['fdbscan-dense-openmp'] = executables['fdbscan-openmp']
executables['fdbscan-hip'] = './arborx_hip'
executables['fdbscan-dense-hip'] = executables['fdbscan-hip']
executables['fdbscan-cuda'] = './arborx_cuda'
executables['fdbscan-dense-cuda'] = executables['fdbscan-cuda']
executables['tepp'] = './tepp'

cmds = {}
cmds['fdbscan-openmp'] = '--impl fdbscan --verbose --core-min-size %d --eps %f -binary --filename %s'
cmds['fdbscan-hip'] = cmds['fdbscan-openmp']
cmds['fdbscan-cuda'] = cmds['fdbscan-openmp']
cmds['fdbscan-dense-openmp'] = '--impl fdbscan-densebox --verbose --core-min-size %d --eps %f --binary --filename %s'
cmds['fdbscan-dense-hip'] = cmds['fdbscan-dense-openmp']
cmds['fdbscan-dense-cuda'] = cmds['fdbscan-dense-openmp']
cmds['tepp'] = '-minpts %d -eps %f %s'


dataset_dir="~/csc333_orion/"
#  dataset_dir = "/datasets"
dataset_dir = dataset_dir + "/"
datasets = [ "2D-ngsim", "2D-porto", "2D-ss-sim", "2D-ss-var", "3D-hacc", "3D-ss-sim", "3D-ss-var", "5D-ss-sim", "5D-ss-var", "7D-ss-sim", "7D-ss-var", "7D-household" ]
filenames = {}
filenames["2D-ngsim"] = dataset_dir + "ngsim.arborx"
filenames["2D-porto"] = dataset_dir + "PortoTaxi.arborx"
filenames["3D-hacc"] = dataset_dir + "hacc_37M.arborx"
filenames["2D-ss-sim"] = dataset_dir + "2D_VisualSim_10M.arborx"
filenames["2D-ss-var"] = dataset_dir + "2D_VisualVar_10M.arborx"
filenames["3D-ss-sim"] = dataset_dir + "3D_VisualSim_10M.arborx"
filenames["3D-ss-var"] = dataset_dir + "3D_VisualVar_10M.arborx"
filenames["5D-ss-sim"] = dataset_dir + "5D_VisualSim_10M.arborx"
filenames["5D-ss-var"] = dataset_dir + "5D_VisualVar_10M.arborx"
filenames["7D-ss-sim"] = dataset_dir + "7D_VisualSim_10M.arborx"
filenames["7D-ss-var"] = dataset_dir + "7D_VisualVar_10M.arborx"
filenames["7D-household"] = dataset_dir + "7D_2.05M_household.arborx"

default_minpts = 10

default_eps = { key:1000 for key in datasets }
default_eps["2D-ngsim"] = 1.0
default_eps["2D-porto"] = 0.005
default_eps["3D-hacc"] = 0.042
default_eps["7D-household"] = 2.0

log_dir = 'perf_portability/'
subprocess.run('mkdir -p ' + log_dir, shell=True)
for dataset in datasets:
    eps = default_eps[dataset]
    minpts = default_minpts
    for code in codes:
        if ((dataset == '2D-ngsim' and re.search('dense', code) != None) or \
            (dataset == '2D-porto' and re.search('fdbscan', code) != None and re.search('dense', code) == None)):
            # Cannot run FDBSCAN-DenseBox on NGSIM
            # Cannot run FDBSCAN on Porto
            continue
        print("%s %s %d %.5f" % (dataset, code, minpts, eps))
        cmd = executables[code] + " " + (cmds[code] % (minpts, eps, filenames[dataset]))
        filename = log_dir + '/' + ("%s_%s_minpts=%d_eps%f.log" % (dataset, code, minpts, eps))
        with open(filename, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
