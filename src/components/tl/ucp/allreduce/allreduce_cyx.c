#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/ec/ucc_ec.h"

// TODO(cyx): 注意这里不要出现 UCS_OK!
// TODO(cyx): progress调用了太多次，可以少一点

#define ALLREDUCE_CYX_PHASE_IDLE      0
#define ALLREDUCE_CYX_PHASE_PUTING    1
#define ALLREDUCE_CYX_PHASE_ATOMICING 2
// #define ALLREDUCE_CYX_PHASE_REDUCING  3
// reduce 不应该放到 task->allreduce_cyx.phase 里。reduce_task 非空自然就是有 reduce 任务

ucc_status_t ucc_tl_ucp_allreduce_cyx_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *team,
                                           ucc_coll_task_t     **task_h)
{
    // fprintf(stderr, "holy shit running my allreduce!\n");
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    // 仿照 alltoall_oneside 检查 global_work_buffer
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    // 仿照我修改过后的 alltoall_oneside 检查 是否已经被映射为适合单边内存的
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    if (UCC_IS_INPLACE(coll_args->args)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "inplace allreduce_cyx are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (UCC_IS_PERSISTENT(coll_args->args)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "persistent allreduce_cyx are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }

    // 创建 task，设置 super（伪父类），标记 start、progress、finalize 函数
    task    = ucc_tl_ucp_init_task(coll_args, team);
    *task_h = &task->super;
    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_allreduce_cyx_start;
    task->super.progress = ucc_tl_ucp_allreduce_cyx_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_cyx_finalize;

    status = UCC_OK;
out:
    return status;
}

// post 一个 coll 的时候会调用这个
// 进行第一次 src 到 dst 的拷贝
// 在 task 中标记好初始化信息
// 把任务提交下去
ucc_status_t ucc_tl_ucp_allreduce_cyx_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         mpi_size  = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         mpi_rank  = task->subset.myrank;
    ucc_rank_t         peer_rank = (mpi_rank + 1) % mpi_size;
    ptrdiff_t          sbuf      = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dbuf      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ptrdiff_t scratch_buf = (ptrdiff_t)TASK_ARGS(task).cyx_scratch.info.buffer;
    ucc_memory_type_t mem_type  = TASK_ARGS(task).dst.info.mem_type;
    size_t            count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t    dt        = TASK_ARGS(task).dst.info.datatype;
    size_t            data_size = count * ucc_dt_size(dt);
    volatile long    *pSync =
        TASK_ARGS(task).global_work_buffer; // 本 rank 已经被几个人完成了 put
    ucc_status_t status;
    size_t       xsegment_size =
        data_size /
        mpi_size; // 每次交换（PUT）的长度。sbuf/dbuf 被我分成很多个 xsegment

    if (count % mpi_size) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "allreduce_cyx count %lu is not multiple of mpi_size %d",
                 count, mpi_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    // 初始化 task 状态
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->onesided.put_posted    = 0;
    task->onesided.put_completed = 0;

    // start 阶段进行第一次的内存拷贝
    // 前 mpi_size - 1 次 PUT（PUT 不在这里）从 scratch 出发到达 dst
    // 后 mpi_size - 1 次 PUT（PUT 不在这里）从 dst 出发到达 dst
    ptrdiff_t copy_start_src     = sbuf + xsegment_size * mpi_rank;
    ptrdiff_t copy_start_scratch = scratch_buf + xsegment_size * mpi_rank;
    ptrdiff_t put_start_dst      = dbuf + xsegment_size * mpi_rank;
    UCC_CHECK_GOTO(ucc_mc_memcpy((void *)copy_start_scratch,
                                 (void *)copy_start_src, data_size / mpi_size,
                                 mem_type, mem_type),
                   out, status);

    // 获得 reduce 任务的执行器
    UCC_CHECK_GOTO(
        ucc_coll_task_get_executor(&task->super, &task->allreduce_cyx.executor),
        out, status);

    // 进行第一次 PUT
    status =
        ucc_tl_ucp_put_nb((void *)put_start_dst, (void *)copy_start_scratch,
                          xsegment_size, peer_rank, team, task);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "ucc_tl_ucp_put_nb failed!");
        goto out;
    }

    // 拷贝完以后，pSync 从 0 开始
    pSync[0]                        = 0;
    task->allreduce_cyx.last_pSync  = 0;
    task->allreduce_cyx.phase       = ALLREDUCE_CYX_PHASE_PUTING;
    task->allreduce_cyx.reduce_task = NULL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return status;
}

// progress: if pSync 变了，则进入 reduce 态。
// 否则（也许不是 else？），如果 pSync 还在n左右以内，则如果在 reduce 态，则检查 reduce 状态。如果在 put 态，则检查 put 状态。如果在 put暂时未完成态，什么也不做
// 如果 pSync 还在2n左右以内，则去掉reduce态，其他和上面一样。如果 pSync 到了（到了更高的境界），则完成
// PUTTING-ATOMICING。期间可以发起 reduce 任务，但是不能检查 reduce 是否完成
// 这是因为，我不知道向同一个UCX ep下发ucp_put_nbx，是不是保序的
// 如果保序，可以跟踪 oneside 计数器（有点悬，要确保这个 task 期间不进行其他 oneside 操作）
// 如果不保序，就要 put 完了才能进行atomic，atomic完了才能检查reduce是否完成
// 因为reduce一旦完成，就要开始新一轮put了。
void ucc_tl_ucp_allreduce_cyx_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         mpi_size  = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         mpi_rank  = task->subset.myrank;
    ucc_rank_t         peer_rank = (mpi_rank + 1) % mpi_size;
    ptrdiff_t          sbuf      = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dbuf      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ptrdiff_t scratch_buf = (ptrdiff_t)TASK_ARGS(task).cyx_scratch.info.buffer;
    size_t    count       = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t dt     = TASK_ARGS(task).dst.info.datatype;
    size_t         data_size = count * ucc_dt_size(dt); // sbuf/dbuf 的长度
    size_t         xsegment_size =
        data_size /
        mpi_size; // 每次交换（PUT）的长度。sbuf/dbuf 被我分成很多个 xsegment
    volatile long *pSync =
        TASK_ARGS(task).global_work_buffer; // 本 rank 已经被几个人完成了 put
    ucc_status_t status;

    // pSync 变化，说明前一个人的 PUT 最新到达，并且当前没有在进行的 reduce 任务
    fprintf(stderr, "pSync %ld lastpSync %ld reduce_task %p phase %d\n",
            pSync[0], task->allreduce_cyx.last_pSync,
            task->allreduce_cyx.reduce_task, task->allreduce_cyx.phase);
    if (pSync[0] > task->allreduce_cyx.last_pSync &&
        task->allreduce_cyx.reduce_task == NULL &&
        task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_IDLE) {
        task->allreduce_cyx.last_pSync++;
        int xsegment_id =
            (mpi_rank + task->allreduce_cyx.last_pSync + mpi_size - 2) %
            mpi_size; // 刚刚收到的 xsegment 的 ID
        ptrdiff_t reduce_start_src = sbuf + xsegment_size * xsegment_id;
        ptrdiff_t reduce_start_dst = dbuf + xsegment_size * xsegment_id;
        ptrdiff_t reduce_start_scratch =
            scratch_buf + xsegment_size * xsegment_id;
        if (task->allreduce_cyx.last_pSync <= mpi_size - 1) { // 还没 reduce 完
            // reduce 结果总是在 scratch 中
            status = ucc_dt_reduce(
                (void *)reduce_start_src, (void *)reduce_start_dst,
                (void *)reduce_start_scratch, xsegment_size / ucc_dt_size(dt),
                dt, args, 0, 0, task->allreduce_cyx.executor,
                &task->allreduce_cyx.reduce_task);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
        } else if (task->allreduce_cyx.last_pSync <= 2 * mpi_size - 2) {
            status = ucc_tl_ucp_put_nb((void *)reduce_start_dst,
                                       (void *)reduce_start_dst, xsegment_size,
                                       peer_rank, team, task);
            task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_PUTING;
            if (status < 0) {
                tl_error(UCC_TASK_LIB(task),
                         "allreduce_cyx ucc_tl_ucp_put_nb reduce failed");
                task->super.status = status;
                return;
            }
        }
        return;
    }
    if (task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_PUTING) {
        // TODO(cyx): 检查 putting，完成转 atomic
        int polls = 0;

        if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
            // 转 atomic
            status = ucc_tl_ucp_atomic_inc((void *)pSync, peer_rank, team);
            if (status < 0) {
                tl_error(UCC_TASK_LIB(task),
                         "allreduce_cyx ucc_tl_ucp_atomic_inc failed");
                task->super.status = status;
                return;
            }
            task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_ATOMICING;
            return;
        }
        while (polls++ < task->n_polls) {
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                status = ucc_tl_ucp_atomic_inc((void *)pSync, peer_rank, team);
                if (status < 0) {
                    tl_error(UCC_TASK_LIB(task),
                             "allreduce_cyx ucc_tl_ucp_atomic_inc failed");
                    task->super.status = status;
                    return;
                }
                task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_ATOMICING;
                break;
            }
            ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
        }
        return;
    }
    if (task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_ATOMICING) {
        // TODO(cyx): 检查 atomic，完成转 idle
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
        task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_IDLE;
        return;
    }
    if (task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_IDLE) {
        // TODO(cyx): 检查是否到达  2mpisize - 1。到达则完成。
        // TODO(cyx): 检查 reduce_task 是否完成，完成转 putting
        if (task->allreduce_cyx.reduce_task == NULL &&
            task->allreduce_cyx.last_pSync == 2 * mpi_size - 1) {
            ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
            task->super.status = UCC_OK;
            return;
        }
        if (task->allreduce_cyx.reduce_task == NULL) {
            return;
        }
        status = ucc_ee_executor_task_test(task->allreduce_cyx.reduce_task);
        if (status == UCC_OK) {
            ucc_ee_executor_task_finalize(task->allreduce_cyx.reduce_task);
            task->allreduce_cyx.reduce_task = NULL;
            int xsegment_id =
                (mpi_rank + task->allreduce_cyx.last_pSync + mpi_size - 2) %
                mpi_size;
            ptrdiff_t put_src = scratch_buf + xsegment_size * xsegment_id;
            ptrdiff_t put_dst = dbuf + xsegment_size * xsegment_id;
            status = ucc_tl_ucp_put_nb((void *)put_dst, (void *)put_src,
                                       xsegment_size, peer_rank, team, task);
            task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_PUTING;
        }
        if (status < 0) {
            tl_error(
                UCC_TASK_LIB(task),
                "allreduce_cyx ucc_ee_executor_task_test or put reduce failed");
            task->super.status = status;
            return;
        }
        // TODO(cyx): 是否要完成一个推进atomic？
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    } else {
        tl_error(UCC_TASK_LIB(task), "unknown task->allreduce_cyx.phase");
    }
    // return ;
    // 彻底完成了设置 task->super.status = OK;
}

ucc_status_t ucc_tl_ucp_allreduce_cyx_finalize(ucc_coll_task_t *coll_task)
{
    // fprintf(stderr, "holy shit finalize my allreduce!\n");
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       status;

    status = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task),
                 "failed finalize ucc_tl_ucp_allreduce_cyx_finalize");
    }
    return status;
}