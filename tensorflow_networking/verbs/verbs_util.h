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

#ifndef TENSORFLOW_CONTRIB_VERBS_VERBS_UTIL_H_
#define TENSORFLOW_CONTRIB_VERBS_VERBS_UTIL_H_

#include <string>
#include <sstream>
#include <errno.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <infiniband/verbs.h>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {
int RDMACQNUMS() {
  const char* env_p = std::getenv("RDMA_CQ_NUMS");
  int nums = 1;
  if (env_p != nullptr) {
    std::stringstream ss(env_p);
    ss >> nums;
  }
   LOG(INFO) << "RDMA_CQ_NUMS:" << nums;
  return nums;
}
int RDMACQPOOLSIZE() {
  const char* env_p = std::getenv("RDMA_CQPOOL_SIZE");
  int pool_size = 20;
  if (env_p != nullptr) {
    std::stringstream ss(env_p);
    ss >> pool_size;
  }
  return pool_size;
}

int RDMATENSORBUFFERRATIO() {
  const char* env_p = std::getenv("RDMA_TENSOR_BUFFER_RATIO");
  int ratio = 5;
  if (env_p != nullptr) {
    std::stringstream ss(env_p);
    ss >> ratio;
  }
  LOG(INFO) << "RDMA_TENSOR_BUFFER_RATIO:" << ratio;
  return ratio;
}

int RDMAENABLESENDDRIERN() {
  const char* env_p = std::getenv("RDMASendDriver");
  int send_driver = 0;
  if (env_p != nullptr) {
    std::stringstream ss(env_p);
    ss >> send_driver;
  }
  return send_driver;
}

int RDMACHUNKSIZE() {
  const char* env_p = std::getenv("RDMAChunkSize");
  int chunk_size = 60*1024*1024;
  if (env_p != nullptr) {
    std::stringstream ss(env_p);
    ss >> chunk_size;
  }
  return chunk_size;
}

std::string GetMetaOutput() {
  const char* env_p = std::getenv("RDMAMetaOutput");
  if (env_p != nullptr) {
    return std::string(env_p);
  }
  return "viewfs://hadoop-meituan/user/hadoop-hdp/wuyongyu02/default_output";
}

std::string GetWorkerMetas() {
  /*
  edg_name#size|edg_name#size
  */
  const char* env_p  = std::getenv("RDMAWorkerMetas");
  if (env_p != nullptr) {
    return std::string(env_p);
  }
  return "edge_6389_global_step;0:0#80";
}

} // end namespace
class VerbsUtil {
 public:
  static string AppendStepidToKey(const string& key, int64 step_id);
  static void GetKeyAndStepId(const string& key_with_step_id, string& key,
                              int64& step_id);
};


#define DIVUP(x, y) (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) (DIVUP((x), (y))*(y))

template <typename T>
static inline T align_floor(T v, T align) {
  return v - (v % align);
}

template <typename T>
static inline T align_ceil(T v, T align) {
  return align_floor(v + align - 1, align);
}

static inline size_t ib_allocate_size(size_t size) {
  size_t page_size = 4096;
  return ROUNDUP(size, page_size);
}

static inline void ib_malloc(void** ptr, size_t* allocate_size, size_t size,
                             int minimum_alignment) {
  void* p;
  *allocate_size = size;
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) {
    p = malloc(size);
  } else {
    int err = posix_memalign(&p, minimum_alignment, size);
  }
  *ptr = p;
}

class MemoryAllocator {
 public:
  explicit MemoryAllocator(struct ibv_pd *pd) {
    std::lock_guard<std::mutex> lk(mu_);
    pd_ = pd;
  }

  ~MemoryAllocator() {
    std::lock_guard<std::mutex> lk(mu_);
    for(auto &it : mr_) {
      ibv_dereg_mr(it.second);
      free(it.first);
    }
  }

  char *Alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }

    // align to page size (usually 4KB)
    size = align_ceil(size, pagesize_);

    char *p;
    size_t allocate_size = size;
    ib_malloc((void**) &p, &allocate_size, size, 64);
    CHECK(p);

    struct ibv_mr *mr;
    CHECK(mr = ibv_reg_mr(pd_, p, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

    std::lock_guard<std::mutex> lk(mu_);
    mr_[p] = mr;
    used_list.emplace(p, size);

    return p;
  }

  uint32_t LocalKey(char *addr) {
    return Addr2MR(addr)->lkey;
  }

  uint32_t RemoteKey(char *addr) {
    return Addr2MR(addr)->rkey;
  }

  struct ibv_pd* GetPD() {
    return pd_;
  }

 private:
  // convert the memory address to its associated RDMA memory region
  inline struct ibv_mr* Addr2MR(char *addr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mr_.find(addr);
    CHECK(it != mr_.end());

    return it->second;
  }

  std::mutex mu_;
  struct ibv_pd *pd_;
  size_t pagesize_ = sysconf(_SC_PAGESIZE);
  std::unordered_map<char *, size_t> used_list;
  std::unordered_map<char *, struct ibv_mr *> mr_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_VERBS_VERBS_UTIL_H_
