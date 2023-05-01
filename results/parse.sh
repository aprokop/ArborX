#!/usr/bin/env bash
arborx_cuda_dir="padirac/"
arborx_hip_dir="crusher/"
arborx_openmp_dir="crusher/"
tepp_dir="crusher/"


echo "Dataset\ttepp\topenmp\thip\tcuda"
for dataset in "2D-porto" "2D-ss-sim" "2D-ss-var" "3D-hacc" "3D-ss-sim" "3D-ss-var" "5D-ss-sim" "5D-ss-var" "7D-ss-sim" "7D-ss-var" "7D-household"; do
    n=$(cat ${arborx_cuda_dir}/${dataset}_fdbscan-dense-cuda*.log | grep "Read in" | cut -f 3 -d ' ')

    arborx_cuda_time=$(cat ${arborx_cuda_dir}/${dataset}_fdbscan-dense-cuda*.log | grep "total time" | cut -f 2 -d ':')
    arborx_cuda_rate=$(echo "scale=1;$n/$arborx_cuda_time/1000000" | bc)

    arborx_hip_time=$(cat ${arborx_hip_dir}/${dataset}_fdbscan-dense-hip*.log | grep "total time" | cut -f 2 -d ':')
    arborx_hip_rate=$(echo "scale=1;$n/$arborx_hip_time/1000000" | bc)

    arborx_openmp_time=$(cat ${arborx_openmp_dir}/${dataset}_fdbscan-dense-openmp*.log | grep "total time" | cut -f 2 -d ':')
    arborx_openmp_rate=$(echo "scale=1;$n/$arborx_openmp_time/1000000" | bc)

    tepp_time=$(cat ${tepp_dir}/${dataset}_tepp*.log | grep "totaltime" | cut -f 2 -d '*')
    tepp_rate=$(echo "scale=1;$n/$tepp_time/1000000" | bc)

    echo "$dataset $tepp_rate $arborx_openmp_rate $arborx_hip_rate $arborx_cuda_rate"
done
