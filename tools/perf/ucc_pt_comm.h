/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COMM_H
#define UCC_PT_COMM_H

#include <ucc/api/ucc.h>
#include "ucc_pt_config.h"
#include "ucc_pt_bootstrap.h"
#include "ucc_pt_bootstrap_mpi.h"

extern "C" {
#include <components/ec/ucc_ec.h>
#include <components/mc/ucc_mc.h>
}

ucc_status_t ucc_pt_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type);

ucc_status_t ucc_pt_free(ucc_mc_buffer_header_t *h_ptr);

class ucc_pt_comm {
    ucc_pt_comm_config      cfg;
    ucc_lib_h               lib;
    ucc_context_h           context;
    ucc_team_h              team;
    void                   *stream;
    ucc_ee_h                ee;
    ucc_ee_executor_t      *executor;
    ucc_pt_bootstrap       *bootstrap;
    ucc_mc_buffer_header_t *send_header;
    ucc_mc_buffer_header_t *recv_header;
    ucc_mc_buffer_header_t *global_work_buffer_header;
    void                    set_gpu_device();

  public:
    ucc_pt_comm(ucc_pt_comm_config config);
    int get_rank();
    int get_size();
    int get_isoneside(); // cyx add
    void set_send_recv_gwb_header(ucc_mc_buffer_header_t **send_hdr,
                                 ucc_mc_buffer_header_t **recv_hdr,
                                 ucc_mc_buffer_header_t **gwb_hdr); // cyx add
    ucc_ee_executor_t *get_executor();
    ucc_ee_h           get_ee();
    ucc_team_h         get_team();
    ucc_context_h      get_context();
    ~ucc_pt_comm();
    ucc_status_t init();
    ucc_status_t barrier();
    ucc_status_t allreduce(double *in, double *out, size_t size,
                           ucc_reduction_op_t op);
    ucc_status_t finalize();
};

#endif
