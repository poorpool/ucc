#!/bin/bash

export OMPI_DIR=/home/cyx/chores/tmpinstall
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

NUMBER_OF_NODES=2
MIN_MESSAGE_SIZE=16384 # in bytes
MAX_MESSAGE_SIZE=16777216 # 16777216
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
MAX_ONESIDE_BUFFER_SIZE=$MAX_MESSAGE_SIZE # 疑似应该除以2

/home/cyx/chores/tmpinstall/bin/mpirun -np $NUMBER_OF_NODES -hostfile $HOST_FILE \
  -mca pml ucx -mca btl ^vader,tcp,openib,uct \
  -x UCX_TLS=rc \
  -x UCX_UD_MLX5_TIMEOUT=2m \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x PATH=$PATH \
  hostname
# -c alltoall -b 16384 -e 16777216
#  -O $MAX_ONESIDE_BUFFER_SIZE
# alltoall 需要自己指定 -O $MAX_ONESIDE_BUFFER_SIZE，但是 allreduce 的 sw 实现不需要
# allreduce只用传 global_work_buffer就行了，他也没检查。。。
# 注意：allreduce不能从很小的就开始设置为 sw 算法。因为 MPI 或者