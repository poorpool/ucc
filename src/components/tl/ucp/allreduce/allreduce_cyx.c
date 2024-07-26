
#include "tl_ucp.h"
#include "allreduce.h"

ucc_status_t ucc_tl_ucp_allreduce_cyx_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *team,
                                           ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    task    = ucc_tl_ucp_init_task(coll_args, team);
    *task_h = &task->super;
    // task->super.post     = ucc_tl_ucp_alltoall_onesided_start;
    // task->super.progress = ucc_tl_ucp_alltoall_onesided_progress;
    status = UCC_OK;
    fprintf(stderr, "holy shit!!! istarted my program\n");
out:
    return status;
}