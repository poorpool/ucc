#!/bin/bash

TOP=/home/cyx/chores/ucc

OMPI_DIR=/home/cyx/chores/tmpinstall
# DPU_HOST_FILE=<dpu hostfile>

# NUMBER_OF_NODES=$(cat $DPU_FILE | grep -v '#' | wc -l)

export UROM_PLUGIN_PATH=<path to ucc doca_plugins>
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

export UCC_DIR=<ucc>
export UCX_DIR=<ucx>
export LD_LIBRARY_PATH=$UCC_DIR/lib:$UCX_DIR/lib:$LD_LIBRARY_PATH

options="-x UCX_TLS=rc_x,tcp $options"

mpirun --tag-output -np $NUMBER_OF_NODES -hostfile $DPU_HOST_FILE -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $options -x UROM_PLUGIN_PATH=$UROM_PLUGIN_PATH $TOP/doca/build-dpu/services/urom/doca_urom_daemon -l 10 --sdk-log-level 10