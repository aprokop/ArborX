#!/usr/bin/env python3

import subprocess


num_threads = "56"

#  codes = [ "fdbscan", "fdbscan-dense", "tepp", "pdsdbscan", "gdbscan", "gowanlock" ]
# codes = [ "fdbscan", "fdbscan-dense", "tepp", "pdsdbscan" ]
codes = [ "fdbscan", "fdbscan-dense", "g-dbscan" ]

# sweeps = [ 'eps', 'minpts', 'size' ]
sweeps = [ 'size' ]

executables = {}
executables["fdbscan"] = "./arborx_cuda"
executables["fdbscan-dense"] = "./arborx_cuda"
executables["tepp"] = "./tepp"
executables["pdsdbscan"] = "OMP_NUM_THREADS=" + num_threads + " ./pdsdbscan-s"
executables["g-dbscan"] = "./g-dbscan"

cmds = {}
cmds["fdbscan"] = "--impl fdbscan --verbose --core-min-size %d --eps %f --samples %d --binary --filename %s"
cmds["fdbscan-dense"] = "--impl fdbscan-densebox --verbose --core-min-size %d --eps %f --samples %d --binary --filename %s"
cmds["tepp"] = "-minpts %d -eps %f -S %d %s"
cmds["pdsdbscan"] = "-m %d -e %f -S %d -t " + num_threads + "-b -i %s"
cmds["g-dbscan"] = "-m %d -e %f -S %d -i %s"


# dataset_dir="~/csc333_orion/"
dataset_dir = "/datasets"
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

default_samples = { key:1000000 for key in datasets }
# Adjust 2D datasets down to run G-DBSCAN
for dataset in ["2D-ngsim", "2D-porto", "2D-ss-sim", "2D-ss-var"]:
    default_samples[dataset] = 100000

default_eps = { key:1000 for key in datasets }
default_eps["2D-ngsim"] = 1.0
default_eps["2D-porto"] = 0.005
default_eps["3D-hacc"] = 0.042
default_eps["7D-household"] = 2.0

epss = { key:[100, 200, 500, 1000, 3000] for key in datasets }
epss["2D-ngsim"] = [ 0.50, 0.75, 1.0, 1.25, 1.50 ]
epss["2D-porto"] = [ 0.002, 0.004, 0.006, 0.008, 0.01 ]
epss["3D-hacc"] = [ 0.01, 0.05, 0.1, 0.5, 1.0 ]
epss["7D-household"] = [0.1, 0.5, 1.0, 2.0, 5.0 ]

cmd = "git --version"

# eps sweep
if 'eps' in sweeps:
    log_dir = 'sweep_eps'
    subprocess.run('mkdir -p ' + log_dir, shell=True)
    for dataset in datasets:
        for eps in epss[dataset]:
            for code in codes:
                if code == "g-dbscan" and dataset[0] != "2":
                    # Skip non-2D datasets for G-DBSCAN
                    continue
                if code == 'pdsdbscan' and dataset == '7D-household' and eps > 1.0:
                    # Getting killed by OOM on Crusher
                    continue
                minpts = default_minpts
                samples = default_samples[dataset]
                print("%s %s %d %.5f %d" % (dataset, code, minpts, eps, samples))
                cmd = executables[code] + " " + (cmds[code] % (minpts, eps, samples, filenames[dataset]))
                filename = log_dir + '/' + ("eps_sweep_%s_%s_minpts=%d_eps%f_samples=%d.log" % (dataset, code, minpts, eps, samples))
                with open(filename, 'w') as f:
                    subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

# minpts sweep
if 'minpts' in sweeps:
    log_dir = 'sweep_minpts'
    subprocess.run('mkdir -p ' + log_dir, shell=True)
    for dataset in datasets:
        for minpts in [10, 50, 100, 500, 1000]:
            for code in codes:
                if code == "g-dbscan" and dataset[0] != "2":
                    # Skip non-2D datasets for G-DBSCAN
                    continue
                if code == 'pdsdbscan' and dataset == '7D-household':
                    # Getting killed by OOM on Crusher
                    continue
                eps = default_eps[dataset]
                samples = default_samples[dataset]
                print("%s %s %d %.5f %d" % (dataset, code, minpts, eps, samples))
                cmd = executables[code] + " " + (cmds[code] % (minpts, eps, samples, filenames[dataset]))
                filename = log_dir + '/' + ("minpts_sweep_%s_%s_minpts=%d_eps%f_samples=%d.log" % (dataset, code, minpts, eps, samples))
                with open(filename, 'w') as f:
                    subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

# size sweep
if 'size' in sweeps:
    log_dir = 'sweep_size'
    subprocess.run('mkdir -p ' + log_dir, shell=True)
    for dataset in datasets:
        for samples in [2**13, 2**15, 2**17, 2**19, 2**21, 2**23]:
            if dataset == '7D-household' and samples > 2049280:
                continue
            for code in codes:
                if code == "g-dbscan" and (dataset[0] != "2" or \
                        (dataset == '2D-ngsim' and samples > 2**23) or \
                        (dataset == '2D-porto' and samples > 2**18) or \
                        (dataset == '2D-ss-sim' and samples > 2**19) or \
                        (dataset == '2D-ss-var' and samples > 2**20)):
                    # Skip non-2D or OOM datasets for G-DBSCAN
                    continue
                if code == 'pdsdbscan' and ((dataset == '7D-household' and samples > 2**19) or \
                        (dataset == '2D-porto' and samples > 2**22) or \
                        (dataset == '2D-ss-sim' and samples > 2**22) or \
                        (dataset == '2D-ss-var' and samples > 2**22)):
                    # Getting killed by OOM on Crusher
                    continue
                minpts = default_minpts
                eps = default_eps[dataset]
                print("%s %s %d %.5f %d" % (dataset, code, minpts, eps, samples))
                cmd = executables[code] + " " + (cmds[code] % (minpts, eps, samples, filenames[dataset]))
                filename = log_dir + '/' + ("size_sweep_%s_%s_minpts=%d_eps%f_samples=%d.log" % (dataset, code, minpts, eps, samples))
                with open(filename, 'w') as f:
                    subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
