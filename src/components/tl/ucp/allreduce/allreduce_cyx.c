#include "allreduce.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/ec/ucc_ec.h"

// TODO(cyx): progress调用了太多次，可以少一点

#define ALLREDUCE_CYX_PHASE_IDLE    0
#define ALLREDUCE_CYX_PHASE_PUTTING 1
// #define ALLREDUCE_CYX_PHASE_ATOMICING 2
// #define ALLREDUCE_CYX_PHASE_REDUCING  3
// reduce 不应该放到 task->allreduce_cyx.phase 里。reduce_task 非空自然就是有 reduce 任务

ucc_status_t ucc_tl_ucp_allreduce_cyx_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *team,
                                           ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    // 要求 global_work_buffer 存在，并且在外部被置为 0！
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    // 仿照我修改过后的 alltoall_oneside 检查是否已经被映射为适合单边操作的内存
    // 并不会真的检查有没有注册，只是看用户有没有意识到
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    // 简单起见，暂不支持 inplace 操作
    if (UCC_IS_INPLACE(coll_args->args)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "inplace allreduce_cyx are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    // 简单起见，暂不支持 persistent 操作
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

// post 一个集合操作的会调用这个
// 1. 主动发起一次 xsegment put
// 2. 在 task 中标记好初始化信息
// 3. ucc_progress_queue_enqueue
ucc_status_t ucc_tl_ucp_allreduce_cyx_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         mpi_size = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         mpi_rank = task->subset.myrank;
    ucc_rank_t peer_rank = (mpi_rank + 1) % mpi_size; // 始终向下个节点发送
    ptrdiff_t      sbuf = (ptrdiff_t)TASK_ARGS(task).src.info.buffer; // src buf
    ptrdiff_t      dbuf = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer; // dst buf
    size_t         count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t dt        = TASK_ARGS(task).dst.info.datatype;
    size_t         data_size = count * ucc_dt_size(dt);
    size_t xsegment_size = data_size / mpi_size; // 每次交换（PUT）的长度

    ucc_status_t status;

    if (count % mpi_size || mpi_size < 2) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "allreduce_cyx count %lu is not multiple of mpi_size %d or "
                 "mpi_size < 2",
                 count, mpi_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    // 初始化 task 状态
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    // 这个在 ucc_tl_ucp_put_nb 中就会被修改，所以需要现在就设置
    task->onesided.put_posted    = 0;
    task->onesided.put_completed = 0;

    // PUT 的描述从 1 开始，也就是没有第 0 次 PUT
    // 第 1 次 PUT（PUT 完接收端 pSync[0] == 1）：从 src 到 dst
    // 第 [2, mpi_size-1] 次 PUT：从 scratch 到 dst
    // 第 [mpi_size, 2*mpi_size-2]次 PUT：从 dst 到 dst
    ptrdiff_t put_src_by_src = sbuf + xsegment_size * mpi_rank;
    ptrdiff_t put_dst_by_dst = dbuf + xsegment_size * mpi_rank;

    // 获得 reduce 任务的执行器
    status =
        ucc_coll_task_get_executor(&task->super, &task->allreduce_cyx.executor);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "ucc_coll_task_get_executor failed!");
        goto out;
    }

    // 进行第一次 PUT
    status = ucc_tl_ucp_put_nb((void *)put_dst_by_dst, (void *)put_src_by_src,
                               xsegment_size, peer_rank, team, task);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "ucc_tl_ucp_put_nb failed!");
        goto out;
    }

    // 注意：这里不能设置 pSync[0] = 0！
    // 因为其他进程可能已经在 PUT 本进程了
    // 应当在外部确保 global_work_buffer 为 0
    task->allreduce_cyx.last_pSync = 0; // 已经感知并处理完了几个 PUT
    task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_PUTTING; // 处于 PUTTING 态
    task->allreduce_cyx.reduce_task = NULL; // 当前没有 reduce task 在进行

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return status;
}

// progress: 探查是否完成。完成时设置 task->super.status = UCC_OK
// put、atomic、reduce 操作不允许同时进行，以简化代码
void ucc_tl_ucp_allreduce_cyx_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         mpi_size = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         mpi_rank = task->subset.myrank;
    ucc_rank_t peer_rank = (mpi_rank + 1) % mpi_size; // 始终向下个节点发送
    ptrdiff_t sbuf = (ptrdiff_t)TASK_ARGS(task).src.info.buffer; // src buf
    ptrdiff_t dbuf = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer; // dst buf
    ptrdiff_t scratch_buf =
        (ptrdiff_t)TASK_ARGS(task).cyx_scratch.info.buffer; // 保存 reduce 结果
    size_t         count = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t dt    = TASK_ARGS(task).dst.info.datatype;
    size_t data_size     = count * ucc_dt_size(dt); // sbuf or dbuf 的长度
    size_t xsegment_size = data_size / mpi_size; // 每次交换（PUT）的长度
    volatile long *pSync =
        TASK_ARGS(task).global_work_buffer; // 本 rank 已经被几个人完成了 put

    ucc_status_t status;

    if (task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_PUTTING) {
        // 检查 task->n_polls + 1 次
        // 完成了就执行 atomic 操作并进入 IDLE 态
        // 没完成就没完成，下次再来
        for (int polls = 0; polls <= task->n_polls; polls++) {
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                status = ucc_tl_ucp_atomic_inc_block(
                    (void *)pSync, peer_rank, team); // 自己写的阻塞原子操作
                if (status < 0) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "allreduce_cyx ucc_tl_ucp_atomic_inc_block failed");
                    task->super.status = status;
                    return;
                }
                task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_IDLE;
                break;
            }
            ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
        }
        return;
    }

    if (task->allreduce_cyx.phase == ALLREDUCE_CYX_PHASE_IDLE) {
        // 还有 reduce 任务，则检查 reduce 是否完成
        if (task->allreduce_cyx.reduce_task != NULL) {
            status = ucc_ee_executor_task_test(task->allreduce_cyx.reduce_task);
            if (status == UCC_OK) {
                // 完成则释放 reduce 任务
                ucc_ee_executor_task_finalize(task->allreduce_cyx.reduce_task);
                task->allreduce_cyx.reduce_task = NULL;
                int xsegment_id =
                    (mpi_rank + task->allreduce_cyx.last_pSync + mpi_size - 2) %
                    mpi_size;
                // 并发起 PUT
                ptrdiff_t put_src_by_scratch =
                    scratch_buf + xsegment_size * xsegment_id;
                ptrdiff_t put_dst_by_dst = dbuf + xsegment_size * xsegment_id;
                status                   = ucc_tl_ucp_put_nb(
                                      (void *)put_dst_by_dst, (void *)put_src_by_scratch,
                                      xsegment_size, peer_rank, team, task);
                task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_PUTTING;
                // 如果是最后一个 reduce 任务，则还要拷贝自己的到 dst
                if (task->allreduce_cyx.last_pSync == mpi_size - 1) {
                    ucc_mc_memcpy((void *)put_dst_by_dst,
                                  (void *)put_src_by_scratch, xsegment_size,
                                  TASK_ARGS(task).dst.info.mem_type,
                                  TASK_ARGS(task).src.info.mem_type);
                }
            }
            if (status < 0) {
                tl_error(UCC_TASK_LIB(task),
                         "allreduce_cyx ucc_ee_executor_task_test or "
                         "ucc_tl_ucp_put_nb failed");
                task->super.status = status;
                return;
            }
            // ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
            return;
        }
        // 此时没有 reduce 任务，则检查是否 pSync 变化，变化说明前一个人的 PUT 最新到达
        if (pSync[0] > task->allreduce_cyx.last_pSync) {
            task->allreduce_cyx.last_pSync++;
            int xsegment_id =
                (mpi_rank + task->allreduce_cyx.last_pSync + mpi_size - 2) %
                mpi_size; // 刚刚收到的 xsegment 的 ID
            ptrdiff_t reduce_start_src = sbuf + xsegment_size * xsegment_id;
            ptrdiff_t reduce_start_dst = dbuf + xsegment_size * xsegment_id;
            ptrdiff_t reduce_start_scratch =
                scratch_buf + xsegment_size * xsegment_id;
            if (task->allreduce_cyx.last_pSync <=
                mpi_size - 1) { // 需要触发 reduce
                // reduce 结果总是在 scratch 中
                status = ucc_dt_reduce((void *)reduce_start_src,
                                       (void *)reduce_start_dst,
                                       (void *)reduce_start_scratch,
                                       xsegment_size / ucc_dt_size(dt), dt,
                                       args, 0, 0, task->allreduce_cyx.executor,
                                       &task->allreduce_cyx.reduce_task);
                if (status != UCC_OK) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to perform dt reduction");
                    task->super.status = status;
                    return;
                }
            } else if (task->allreduce_cyx.last_pSync <
                       2 * mpi_size - 2) { // 不需要 reduce 但是需要传递 PUT
                status = ucc_tl_ucp_put_nb(
                    (void *)reduce_start_dst, (void *)reduce_start_dst,
                    xsegment_size, peer_rank, team, task);
                task->allreduce_cyx.phase = ALLREDUCE_CYX_PHASE_PUTTING;
                if (status < 0) {
                    tl_error(UCC_TASK_LIB(task),
                             "allreduce_cyx ucc_tl_ucp_put_nb reduce failed");
                    task->super.status = status;
                    return;
                }
            } else if (task->allreduce_cyx.last_pSync ==
                       2 * mpi_size -
                           2) { // 我的任务完成啦（所有 PUT、atomic、reduce）
                task->super.status = UCC_OK;
                return;
            }
            return;
        }
        return;
    }
}

ucc_status_t ucc_tl_ucp_allreduce_cyx_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       status;

    status = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task),
                 "failed finalize ucc_tl_ucp_allreduce_cyx_finalize");
    }
    return status;
}