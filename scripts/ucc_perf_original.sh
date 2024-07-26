#!/bin/bash

export OMPI_DIR=/home/cyx/chores/tmpinstall
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

export NUMBER_OF_NODES=1
export HOST_FILE=/home/cyx/chores/rdma-competition/mpi_hosts

mpirun -np $NUMBER_OF_NODES -hostfile $HOST_FILE \
  -mca pml ucx -mca btl ^vader,tcp,openib,uct \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x OMPI_MCA_coll_ucc_enable=1 \
  -x OMPI_MCA_coll_ucc_priority=100 \
  ucc_perftest -c allreduce -b 16384 -e 16777216
