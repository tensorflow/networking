/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_
#define TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_

#ifdef TENSORFLOW_USE_VERBS

#include <string>
#include <unordered_map>

#include "tensorflow_networking/verbs/rdma.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
// For timeline logger
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"

namespace tensorflow {

class RdmaChannel;
class RdmaAdapter;
class RdmaTensorRequest;
class RdmaMgr {
  friend class RdmaChannel;
  friend class RdmaAdapter;
  friend class RdmaSendDriverMgr;

 public:
  explicit RdmaMgr(const WorkerEnv* const worker_env,
                   GrpcChannelCache* const channel_cache);
  ~RdmaMgr();
  RdmaChannel* FindChannel(const string& key);
  void SetupChannels();
  bool ConnectivityCheck();
  void InitAllocators();
  static void RegMemVisitors();
  const string& local_worker() { return local_worker_; }

  bool NotifyAsyncAllocator();

  bool NotifyAsyncAllocatorTest();

 public:
  string local_worker_;
  const WorkerEnv* const worker_env_;
  GrpcChannelCache* const channel_cache_;

 private:
  size_t num_remote_workers_;
  RdmaAdapter* rdma_adapter_;
  typedef std::unordered_map<string, RdmaChannel*> ChannelTable;
  ChannelTable channel_table_;
  TF_DISALLOW_COPY_AND_ASSIGN(RdmaMgr);
};

class RdmaBasicCPUAllocator : public SubAllocator {
 public:
  RdmaBasicCPUAllocator(const std::vector<SubAllocator::Visitor>& alloc_visitors,
      const std::vector<SubAllocator::Visitor>& free_visitors) :
      SubAllocator(alloc_visitors, free_visitors) {
    numa_node_ = port::kNUMANoAffinity;
  }

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      if (numa_node_ == port::kNUMANoAffinity) {
        ptr = port::AlignedMalloc(num_bytes, static_cast<int>(alignment));
      } else {
        ptr =
          port::NUMAMalloc(numa_node_, num_bytes, static_cast<int>(alignment));
      }
      VisitAlloc(ptr, numa_node_, num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (num_bytes > 0) {
      VisitFree(ptr, numa_node_, num_bytes);
      if (numa_node_ == port::kNUMANoAffinity) {
        port::AlignedFree(ptr);
      } else {
        port::NUMAFree(ptr, num_bytes);
      }
    }
  }

 private:
  int numa_node_;
  TF_DISALLOW_COPY_AND_ASSIGN(RdmaBasicCPUAllocator);
};

// TODO(wuyongyu02): remove this class and its registration when the default
// cpu_allocator() returns visitable allocator
class BFCRdmaAllocator : public BFCAllocator {
 public:
  BFCRdmaAllocator(const std::vector<SubAllocator::Visitor>& alloc_visitors,
                  const std::vector<SubAllocator::Visitor>& free_visitors)
      : BFCAllocator(new RdmaBasicCPUAllocator(alloc_visitors,
                      free_visitors), 1LL << 36, true, "cpu_rdma_bfc") {
  }
};
// REGISTER_MEM_ALLOCATOR("BFCRdmaAllocator", 101, BFCRdmaAllocator);

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // TENSORFLOW_CONTRIB_VERBS_RDMA_MGR_H_
