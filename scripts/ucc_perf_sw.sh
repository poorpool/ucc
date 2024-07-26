#!/bin/bash

export OMPI_DIR=/home/cyx/chores/tmpinstall
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

NUMBER_OF_NODES=2
MIN_MESSAGE_SIZE=16384 # in bytes
MAX_MESSAGE_SIZE=16384 # 16777216
HOST_FILE=/home/cyx/chores/ucc/scripts/mpi_hosts

# 计算总 count
if (( MIN_MESSAGE_SIZE % 4 != 0 )); then
  echo "Error: MIN_MESSAGE_SIZE is not a multiple of 4"
  exit 1
else
  MIN_COUNTS=$((MIN_MESSAGE_SIZE / 4))
fi
if (( MAX_MESSAGE_SIZE % 4 != 0 )); then
  echo "Error: MAX_MESSAGE_SIZE is not a multiple of 4"
  exit 1
else
  MAX_COUNTS=$((MAX_MESSAGE_SIZE / 4))
fi
if (( MAX_MESSAGE_SIZE % NUMBER_OF_NODES != 0 )); then
  echo "Error: MAX_MESSAGE_SIZE is not a multiple of NUMBER_OF_NODES"
  exit 1
else
  MAX_ONESIDE_BUFFER_SIZE=$((MAX_MESSAGE_SIZE / NUMBER_OF_NODES))
fi

mpirun -np $NUMBER_OF_NODES -hostfile $HOST_FILE \
  -mca pml ucx -mca btl ^vader,tcp,openib,uct \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x OMPI_MCA_coll_ucc_enable=1 \
  -x OMPI_MCA_coll_ucc_priority=100 \
  -x UCC_TL_UCP_TUNE="alltoall:0-inf:@1" \
  ucc_perftest -c alltoall -n 10 -b $MIN_COUNTS -e $MAX_COUNTS -O $MAX_ONESIDE_BUFFER_SIZE -d float32
# -c alltoall -b 16384 -e 16777216