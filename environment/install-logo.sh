#!/usr/bin/env bash

conda_env=krill
source activate ${conda_env}

kernel_dir=$(jupyter kernelspec list | grep ${conda_env} | awk '{print $2}')

cp -v logo*.png ${kernel_dir}/.