#!/usr/bin/env bash

conda_env=$(grep name: environment.yml | awk '{print $2}')

source activate ${conda_env}

kernel_dir=$(jupyter kernelspec list | grep ${conda_env} | awk '{print $2}')

rm -f ${kernel_dir}/logo*.png
cp -v logo*.png ${kernel_dir}/.