#export KMP_BLOCKTIME=1
#export KMP_SETTINGS=1
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

#export CCL_ALLREDUCE=recursive_doubling
export CCL_PROCESS_LAUNCHER=none

export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
#export CCL_ITT_LEVEL=1
export CCL_WORKER_COUNT=1

#if turn this line on, need to use a small iteration count such as 50
#export CCL_SCHED_PROFILE=1

#for 48 core *2
#set CCL_WORKER_AFFINITY if necessary
#export CCL_WORKER_AFFINITY=10,22,34,46,58,70,82,94

#export TORCH_COMPILE_DEBUG=1
TORCHINDUCTOR_FREEZING=1 OMP_NUM_THREADS=12 numactl -C 0-11 python inference-test.py --model bigscience/bloom-3b --batch_size 1 --dtype float32 --hf_baseline --test_performance
#deepspeed --num_gpus 1 --bind_cores_to_rank --bind_core_list 0-11 inference-test.py --model bigscience/bloom-3b --batch_size 1 --dtype bfloat16
