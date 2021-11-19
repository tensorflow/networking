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

#ifdef TENSORFLOW_USE_VERBS

#include <fcntl.h>
#include <cstdlib>
#include <regex>
#include <bitset>
#include <inttypes.h>
#include <sstream>
#include <utility>
#include <set>

#include "tensorflow_networking/verbs/rdma.h"
#include "tensorflow_networking/verbs/verbs_service.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

#define RoCE_V2 "RoCE v2"

namespace {

// convenience function for printing message
string MessageTypeToString(RdmaMessageType rmt) {
  switch (rmt) {
    case RDMA_MESSAGE_META_DATA_UPDATE:
      return "RDMA_MESSAGE_META_DATA_UPDATE";
      break;
    case RDMA_MESSAGE_TENSOR_RE_REQUEST:
      return "RDMA_MESSAGE_TENSOR_RE_REQUEST";
      break;
    case RDMA_MESSAGE_TENSOR_REQUEST:
      return "RDMA_MESSAGE_TENSOR_REQUEST";
      break;
    case RDMA_MESSAGE_DRIVER_BEGIN:
      return "RDMA_MESSAGE_DRIVER_BEGIN";
      break;
    case RDMA_MESSAGE_ERROR_STATUS:
      return "RDMA_MESSAGE_ERROR_STATUS";
      break;
    default:
      return "UNKNOWN MESSAGE";
  }
}
}  // namespace

// Function to get environment variable
// Args:
//    var_name - the name of the environmental variable
// Returns:
//    string with it's value or empty string if not set
string get_env_var(char const* var_name) {
  char const* var_temp = getenv(var_name);

  return (var_temp == NULL) ? string() : string(var_temp);
}

// Function to open device
// Args:
//   ibv_dev device to open
// Returns:
//   context of the opened device
ibv_context* open_device(ibv_device* ibv_dev) {
  ibv_context* context = ibv_open_device(ibv_dev);

  LOG(INFO) << "RDMA context->num_comp_vectors:" << context->num_comp_vectors;

  CHECK(context) << "Open context failed for " << ibv_get_device_name(ibv_dev);
  return context;
}

// Function to count the number of active ports for device
// Args:
//   device - to check active ports
// Returns:
//   number of active ports of the given device
int get_dev_active_port_count(ibv_device* device) {
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  ibv_context* context = NULL;
  int rc, port_index, active_ports = 0;

  context = ibv_open_device(device);
  CHECK(context) << "Open context failed for " << ibv_get_device_name(device);
  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device";
  LOG(INFO) << "[RDMA Device Info] "
            << " max_qp:" << device_att.max_qp
            << " max_cq:" << device_att.max_cq
            << " max_pd:" << device_att.max_pd
            << " max_mr:" << device_att.max_mr
            << " max_mr_size:" << device_att.max_mr_size;


  for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
    rc = ibv_query_port(context, port_index, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_index;
    if (port_attr.state == IBV_PORT_ACTIVE) {
      active_ports++;
    }
  }
  ibv_close_device(context);
  return active_ports;
}

// Function to set device. If RDMA_DEVICE not set, search for device with active
// port.
// Fails if more than one device with active port was found.
// Returns:
//   device to use
ibv_device* set_device() {
  ibv_device** dev_list;
  int dev_num, device_index, device_to_open = 0;
  int num_devs_with_active_port = 0;
  string env_p_rdma_device, str_port_num;

  dev_list = ibv_get_device_list(&dev_num);
  CHECK(dev_list) << "No InfiniBand device found";

  env_p_rdma_device = get_env_var("RDMA_DEVICE");
  if (!env_p_rdma_device.empty()) {
    for (device_index = 0; device_index < dev_num; device_index++) {
      if (!env_p_rdma_device.compare(
              ibv_get_device_name(dev_list[device_index]))) {
        CHECK(get_dev_active_port_count(dev_list[device_index]) != 0)
            << "Device " << ibv_get_device_name(dev_list[device_index])
            << " has no active ports";
        return dev_list[device_index];
      }
    }
    // check validity of input device
    CHECK(false) << "The device " << env_p_rdma_device << " wasn't found";
  } else {
    // set default device
    str_port_num = get_env_var("RDMA_DEVICE_PORT");
    CHECK(str_port_num.empty())
        << "RDMA_DEVICE should be provided if RDMA_DEVICE_PORT is set by user";
    for (device_index = 0; device_index < dev_num; device_index++) {
      // get port_num
      if (get_dev_active_port_count(dev_list[device_index]) > 0) {
        num_devs_with_active_port++;
        CHECK(num_devs_with_active_port <= 1) << ". More than one device with "
                                                 "active port in the system. "
                                                 "Please enter RDMA_DEVICE";
        // found device with at least 1 active port
        device_to_open = device_index;
      }
    }
    CHECK(num_devs_with_active_port > 0)
        << "There is no active port in the system";
    return dev_list[device_to_open];
  }
  CHECK(false) << "No device was set!";
  return NULL;  // never happens
}

// Function to set port for device.
// If RDMA_DEVICE_PORT not set, first active port of the device will be set.
// Args:
//   context of the device
// Returns:
//   port to use
uint8_t set_port(ibv_context* context) {
  uint8_t port_num = 0;  // 0 is illegal port number
  string str_port_num;
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  int rc, port_index;

  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device\n";

  str_port_num = get_env_var("RDMA_DEVICE_PORT");
  // user defined port
  if (!str_port_num.empty()) {
    port_num = stoi(str_port_num);
    CHECK(port_num > 0) << "RDMA_DEVICE_PORT should be positive";
    CHECK(port_num <= device_att.phys_port_cnt) << "RDMA_DEVICE_PORT should be "
                                                   "less or equal to amount of "
                                                   "available ports";
    rc = ibv_query_port(context, port_num, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_num;
    // check if port id active
    CHECK(port_attr.state == IBV_PORT_ACTIVE)
        << "Selected RDMA_DEVICE_PORT is not active";
  } else {  // set default port
    for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
      rc = ibv_query_port(context, port_index, &port_attr);
      CHECK(!rc) << "Failed to query the port" << port_index;
      if (port_attr.state == IBV_PORT_ACTIVE) {
        port_num = port_index;
        break;
      }
    }
    CHECK_GT(port_num, 0) << "No active ports";
  }
  return port_num;
}

// Function read from sysfs file
// Args:
//   dir - directory
//   file - file
//   buff - buffer for the result
//   size - buffer size
// Returns:
//   number of bytes were read or -1 if failed
int read_sysfs_file(const char* dir, const char* file, char* buf, size_t size) {
  char* path;
  int fd;
  int len;

  if (asprintf(&path, "%s/%s", dir, file) < 0) return -1;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    free(path);
    return -1;
  }

  len = read(fd, buf, size);

  close(fd);
  free(path);

  if (len > 0 && buf[len - 1] == '\n') buf[--len] = '\0';

  return len;
}

// Function to check if GID index support RoCE V2
// Args:
//   context - device context
//   port_num - port number
//   index -  GID index
// Returns:
//   if GID supports RoCE V2 - true, otherwise - false.
bool is_gid_type_roce_v2(ibv_context* context, uint8_t port_num,
                         uint8_t index) {
  char name[32];
  char buff[41];

  snprintf(name, sizeof(name), "ports/%d/gid_attrs/types/%d", port_num, index);
  if (read_sysfs_file(context->device->ibdev_path, name, buff, sizeof(buff)) <=
      0) {
    return false;
  }
  return !strcmp(buff, RoCE_V2);
}

// Function to set GID index.
// If the port link is IB, no GID index should be selected.
// If Ethernet but RDMA_GID_INDEX not set gid index that supports
//   RoCE V2 will be chosen(fails if more than one IP is configured)
// Args:
//   context - device context
//   port_num - port number
// Returns:
//   GID index to use
uint8_t set_gid(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  string gid_str;
  int rc, i, gids_num = 0, v2_ip_num = 0;
  union ibv_gid gid;
  uint8_t gid_index = 0;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  for (i = 0; i < port_attr.gid_tbl_len; i++) {
    rc = ibv_query_gid(context, port_num, i, &gid);
    CHECK(!rc) << "Failed to query gid to port " << (int)port_num << " index "
               << i;
    if (gid.global.interface_id) {
      gids_num++;
      if (gid.global.subnet_prefix == 0 &&
          is_gid_type_roce_v2(context, port_num, i)) {
        if (v2_ip_num == 0) {
          // can be overwritten by RDMA_GID_INDEX later
          gid_index = i;
        }
        v2_ip_num++;
      }
    }
  }
  switch (port_attr.link_layer) {
    case (IBV_LINK_LAYER_ETHERNET):
      gid_str = get_env_var("RDMA_GID_INDEX");
      if (!gid_str.empty()) {
        gid_index = stoi(gid_str);
        CHECK(gid_index < gids_num)
            << "RDMA_GID_INDEX should be less than GIDs amount" << gids_num;
      } else {
        CHECK(v2_ip_num <= 1)
            << "More than one IP is available, please specify GID_INDEX";
      }
      break;
    case (IBV_LINK_LAYER_INFINIBAND):  // no need in GID index
      break;
    default:
      LOG(INFO) << "Unknown port link layer. Currently supporting Ethernet and "
                   "InfiniBand only. ";
  }
  if (!is_gid_type_roce_v2(context, port_num, gid_index)) {
    LOG(INFO) << "RoCE v2 is not configured for GID_INDEX " << (int)gid_index;
  }
  return gid_index;
}

// set the default or environment value to the configuration parameter.
// Args:
//   default_val- the default value for this parameter
//   env_param- the environment parameter's name
// Returns:
//   32-bit value
uint32_t set_param(uint32_t default_val, const char* env_param) {
  uint32_t val = default_val;
  string val_s;

  val_s = get_env_var(env_param);

  if (!val_s.empty()) {
    val = stoi(val_s);
  }
  return val;
}

enum ibv_mtu set_mtu(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  enum ibv_mtu mtu = IBV_MTU_512;
  string mtu_s;
  int rc, mtu_i;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  mtu_s = get_env_var("RDMA_MTU");

  if (!mtu_s.empty()) {
    mtu_i = stoi(mtu_s);
    switch (mtu_i) {
      case 256:
        mtu = IBV_MTU_256;
        break;
      case 512:
        mtu = IBV_MTU_512;
        break;
      case 1024:
        mtu = IBV_MTU_1024;
        break;
      case 2048:
        mtu = IBV_MTU_2048;
        break;
      case 4096:
        mtu = IBV_MTU_4096;
        break;
      default:
        CHECK(0) << "Error: MTU input value must be one of the following: 256, "
                    "512, 1024, 2048, 4096. MTU "
                 << mtu << " is invalid\n";
        break;
    }
    CHECK(mtu < port_attr.active_mtu)
        << "MTU configuration for the QPs is larger than active MTU";
  } else {
    mtu = port_attr.active_mtu;
  }
  return mtu;
}

RdmaParams params_init(ibv_context* context) {
  RdmaParams params;

  params.port_num = set_port(context);
  params.sgid_index = set_gid(params.port_num, context);
  params.pkey_index = (uint8_t)set_param(PKEY_DEFAULT, "RDMA_PKEY");
  params.queue_depth = set_param(QUEUE_DEPTH_DEFAULT, "RDMA_QUEUE_DEPTH");
  params.timeout = (uint8_t)set_param(TIMEOUT_DEFAULT, "RDMA_TIMEOUT");
  params.retry_cnt = (uint8_t)set_param(RETRY_CNT_DEFAULT, "RDMA_RETRY_CNT");
  params.sl = (uint8_t)set_param(SL_DEFAULT, "RDMA_SL");
  CHECK(params.sl <= 7) << "SL value is " << (int)params.sl
                        << ". Valid values are 0-7.";
  params.mtu = set_mtu(params.port_num, context);
  params.traffic_class = set_param(TRAFFIC_CLASS, "RDMA_TRAFFIC_CLASS");
  return params;
}

ibv_pd* alloc_protection_domain(ibv_context* context) {
  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK(pd) << "Failed to allocate protection domain";
  return pd;
}

Chunk::Chunk(struct ibv_pd* pd) :
    pd_(pd), allocate_size_(0), curr_size_(0), empty_size_(0),
    offset_(0), total_waste_size_(0), total_realloc_size_(0) {
  chunk_addr_size = VerbsEnvRegistrar::Instance()->RdmaChunkSize();
  if (EIGEN_MAX_ALIGN_BYTES > 0) {
    int ratio = (chunk_addr_size + EIGEN_MAX_ALIGN_BYTES) / EIGEN_MAX_ALIGN_BYTES;
    chunk_addr_size = ratio * EIGEN_MAX_ALIGN_BYTES;
  }
  LOG(INFO) << "chunk size:" 
            << chunk_addr_size 
            << " EIGEN_MAX_ALIGN_BYTES:"
            << EIGEN_MAX_ALIGN_BYTES;
}

void Chunk::FreeChunk() {
  LOG(INFO) << "delete Chunk";
  for (auto& it : mrs_) {
    ibv_dereg_mr(it);
  }
  for (auto& it : chunk_addrs_) {
    free(it);
  }
}

Chunk::~Chunk() { }

void Chunk::Alloc(size_t size, void** p, ibv_mr** mr, size_t realloc_size) {
  mutex_lock l(alloc_mu_);
  size_t align_size = size;
  if (EIGEN_MAX_ALIGN_BYTES > 0) {
    int ratio = (size + EIGEN_MAX_ALIGN_BYTES - 1) / EIGEN_MAX_ALIGN_BYTES;
    align_size = ratio * EIGEN_MAX_ALIGN_BYTES;
  }
  // empty addr need alloc new data
  if (empty_size_ < align_size) {
    size_t malloc_size = (align_size + chunk_addr_size - 1) / chunk_addr_size * chunk_addr_size;
    curr_size_ += malloc_size;
    total_waste_size_ += empty_size_;
    total_realloc_size_ += realloc_size;
    LOG(INFO) << "RDMA Allocate Memory: " << curr_size_ << " Bytes " << total_waste_size_ << " " << total_realloc_size_;
    offset_ = 0;
    empty_size_ = malloc_size;
    size_t allocate_size= 0;
    ib_malloc((void**)&new_p_, &allocate_size, malloc_size, 
              EIGEN_MAX_ALIGN_BYTES);
    new_mr_ = ibv_reg_mr(pd_, new_p_, malloc_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    mrs_.emplace_back(new_mr_);
    chunk_addrs_.emplace_back(new_p_);
  }
  *p = (void*)(((char *)new_p_) + offset_);
  empty_size_ -= align_size;
  *mr = new_mr_;
  offset_ += align_size;
}

RdmaAdapter::RdmaAdapter(const WorkerEnv* worker_env)
    : context_(open_device(set_device())),
      params_(params_init(context_)),
      pd_(alloc_protection_domain(context_)),
      worker_env_(worker_env) {
  recv_chunk_ =  new Chunk(pd_);
  cq_nums_ = VerbsEnvRegistrar::Instance()->RdmaCqNums();
  wc_vec_ = new ibv_wc*[cq_nums_];
  cq_vec_ = new ibv_cq*[cq_nums_];
  event_channel_vec_ = new ibv_comp_channel*[cq_nums_];
  for (int i = 0; i < cq_nums_; i++) {
    wc_vec_[i] = new ibv_wc[MAX_CONCURRENT_WRITES * 2];
    event_channel_vec_[i] = ibv_create_comp_channel(context_);
    CHECK(event_channel_vec_[i]) << "Failed to create of "  << i
                                 << " completion channel";
    cq_vec_[i] = ibv_create_cq(context_, MAX_CONCURRENT_WRITES * 2, NULL,
                               event_channel_vec_[i], 0);
    CHECK(cq_vec_[i]) << "Failed to create of " << i << " completion queue";
    CHECK(!ibv_req_notify_cq(cq_vec_[i], 0))
        << "Failed to request CQ notification";
  }
  LOG(INFO) << "RdmaCQpoolSize:"
            << VerbsEnvRegistrar::Instance()->RdmaCQpoolSize();
  pool_ = new thread::ThreadPool(Env::Default(), ThreadOptions(),
      "process_wr_impl", VerbsEnvRegistrar::Instance()->RdmaCQpoolSize(),
      false, nullptr);
}

RdmaAdapter::~RdmaAdapter() {
  for (int i = 0; i < cq_nums_; i++) {
    polling_thread_vec_[i].reset();
  }
  for (int i = 0; i < cq_nums_; i++) {
    CHECK(!ibv_destroy_cq(cq_vec_[i])) << "Failed to destroy CQ";
    CHECK(!ibv_destroy_comp_channel(event_channel_vec_[i]))
        << "Failed to destroy channel";
  }
  CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD";
  CHECK(!ibv_close_device(context_)) << "Failed to release context";
  recv_chunk_->FreeChunk();
  delete recv_chunk_;
  recv_chunk_ = nullptr;
}

void RdmaAdapter::StartPolling() {
  for (int i = 0; i < cq_nums_; i++) {
    polling_thread_vec_.emplace_back(Env::Default()->StartThread(
      ThreadOptions(), "RdmaAdapterCQThread",
      [this, i] { Pool_Process_CQ(i); }));
  }
  VLOG(2) << "Start RdmaAdapter: " << name();
}

string RdmaAdapter::name() const { return string(context_->device->name); }

void RdmaAdapter::Process_WR(ibv_wc wc_, int cq_num) {
  if (wc_.status != IBV_WC_SUCCESS) {
      return;
    }
  CHECK(wc_.status == IBV_WC_SUCCESS)
      << "Failed status \n"
      << ibv_wc_status_str(wc_.status) << " " << wc_.status << " "
      << static_cast<int>(wc_.wr_id) << " " << wc_.vendor_err;
  if (wc_.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    RdmaChannel* rc = reinterpret_cast<RdmaChannel*>(wc_.wr_id);
    if (rc == nullptr) {
      LOG(FATAL) << "Process_WR Faild wc_.wr_id:" << wc_.wr_id
                 << " cq_num:" << cq_num;
      return;
    }
    // put back a recv wr.
    rc->Recv();
    // imm_data is the index of RX buffer in the buffer table.
    uint32_t imm_data = wc_.imm_data;
    RdmaMessageBuffer* rb;
    RdmaMessage rm;

    if (imm_data > RDMA_IMM_MAX_REQUEST_ID && imm_data <= RDMA_IMM_DATA_ACK) {
      // receive an ack to a message
      int pair_index = imm_data - RDMA_IMM_MAX_REQUEST_ID -1;
      int buffer_index = 2 * pair_index;
      rb = rc->message_buffers()[buffer_index];
      rb->SetBufferStatus(remote, idle);
      rb->SendNextItem();
      return;
    }

    if (imm_data <= RDMA_IMM_MAX_REQUEST_ID) {
      // receive a tensor RDMA write
      uint32_t request_index = imm_data;
      if (imm_data < RDMA_IMM_MIN_SENDMGR_BASE) {
        RdmaTensorRequest* request = rc->GetTensorRequest(request_index);
        if (request == nullptr) {
          LOG(INFO) << "Normal request_index:"
                    << request_index
                    << " , Normal request is done by SendDriverMgr";
          return;
        }
        RDMA_LOG(1) << "DoNormal request_index:" << request_index;
        request->RecvTensorContent();
      } else {
        // RecvSendDriver
        const auto& tensors_uid_parsed_key =
            rc->channel_record_->GetChannelTensorsUidParsedkey();
        const auto& it = tensors_uid_parsed_key.find(imm_data);
        if (it == tensors_uid_parsed_key.end()) {
          LOG(FATAL) << "RdmaTensorRequest Not find parsed_key:"
                    << it->second;
        }
        const auto& parsed_key = it->second;
        bool has_data = false;
        std::shared_ptr<DriverEntry> entry_ptr =
            rc->rdma_send_driver_mgr_->GetDriverEntry(parsed_key, &has_data);

        const auto& tensors_meta_data =
            rc->channel_record_->GetChannelTensorsMetaData();
        const auto& meta = tensors_meta_data.find(parsed_key);
        if (meta == tensors_meta_data.end()) {
          LOG(FATAL)
              << "meta is not find in rc->channel_record_->tensors_meta_data_";
        }

        bool can_memcpy = DataTypeCanUseMemcpy(meta->second.data_type_);
        if (!has_data) {
          // parsed DriverPrefixMessage
          DriverPrefixMessage driver_prefix =
              DriverPrefixMessage::ParseDriverPrefixMessage(
                  (void*)entry_ptr->addr_, meta->second.meta_changed_);
          Tensor* val;
          void* entry_tensor_addr = nullptr;
          // get rama's offset addr of Tensor 
          if (meta->second.meta_changed_) {
            entry_tensor_addr = (void*)(entry_ptr->addr_ +
                DriverPrefixMessage::kPrefixMessageTotalBytes);
          } else {
            entry_tensor_addr = (void*)(entry_ptr->addr_ +
                DriverPrefixMessage::CkPrefixMessageTotalBytes);
          }
          if (can_memcpy) {
            // tensor can use zero-copy
            auto fake_allocator = new FakeAllocator(entry_tensor_addr);
            if (meta->second.meta_changed_) {
              val = new Tensor(fake_allocator, 
                               meta->second.data_type_,
                               driver_prefix.tensor_shape_);
            } else {
              val = new Tensor(fake_allocator, 
                               meta->second.data_type_,
                               meta->second.tensor_shape_);
            }
            // memcpy(DMAHelper::base(val), entry_tensor_addr, val->TotalBytes());
          } else {
            // proto should not used zero-copy
            if (meta->second.meta_changed_) {
              val = new Tensor(meta->second.data_type_,
                               driver_prefix.tensor_shape_);
            } else {
              val = new Tensor(meta->second.data_type_,
                               meta->second.tensor_shape_);
            }
            TensorProto proto;
            CHECK(ParseProtoUnlimited(&proto,entry_tensor_addr,
                                      driver_prefix.tensor_bytes_))
                << " fail to parse proto from array";
            if (proto.dtype() > 0 && proto.dtype() <= DataType_MAX) {
              Tensor parsed(proto.dtype());
              if (parsed.FromProto(cpu_allocator(), proto)) {
                *val = std::move(parsed);
              }
            }
          }
          Status s = Status::OK();
          bool is_dead = driver_prefix.is_dead_;
          int64 recv_micros = 0;
          Rendezvous::Args send_args = Rendezvous::Args();
          rc->local_driver_buffer_mgr_->QueueRdmaSave(parsed_key,
              send_args, val, is_dead, recv_micros);
          // if (val != nullptr) {
          //   delete val;
          //   val = nullptr;
          // }
        } else {
          // When recv a SendDriverData which means that :
          // Localrecv SendDriver is Ready.
          LOG(FATAL) << "Local recv SendDriver Data is not ready"
                    << " has_data:" << has_data;
        }
      }
      return;
    }

    // receive a control message
    int pair_index = imm_data - RDMA_IMM_DATA_ACK -1;
    int buffer_index = 2 * pair_index + 1;
    rb = rc->message_buffers()[buffer_index];
    RdmaMessage::ParseMessage(rm, rb->buffer_);
    RdmaMessageBuffer::SendAck(rc, pair_index+1);
    RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
                << ": Received " << MessageTypeToString(rm.type_) << " "
                << "#" << rm.request_index_ << ": " << rm.name_;
    RDMA_LOG(1) << "pair_index imm_data:" << imm_data
              << " Process_WR rm type:" << MessageTypeToString(rm.type_);

    if (rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) {
      RdmaTensorResponse* response = rc->AddTensorResponse(rm);
      RDMA_LOG(1) << "GetResponse....";
      response->Start();
    } else if (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) {
      RDMA_LOG(1) << "Recevive RDMA_MESSAGE_META_DATA_UPDATE";
      RdmaTensorRequest* request = rc->GetTensorRequest(rm.request_index_);
      if (request == nullptr) {
        LOG(FATAL) << "RDMA_MESSAGE_META_DATA_UPDATE request : "
                  << rm.request_index_ << " is already done by LocalBufferMgr.";
      }
      request->RecvTensorMetaData(rm.data_type_, rm.tensor_shape_,
                                      rm.is_dead_, rm.tensor_bytes_);
#ifdef RDMA_DATA_VALIDATION
      request->RecvTensorChecksum(rm.checksum_);
#endif
    } else if (rm.type_ == RDMA_MESSAGE_DRIVER_BEGIN) {
      LOG(INFO) << "Recevive RDMA_MESSAGE_DRIVER_BEGIN";
      RdmaTensorRequest* request = rc->GetTensorRequest(rm.request_index_);
      if (request == nullptr) {
        LOG(INFO) << "RDMA_MESSAGE_DRIVER_BEGIN request : "
                  << rm.request_index_ << " is already done by LocalBufferMgr.";
      }
    } else if (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST) {
      RdmaTensorResponse* response = rc->UpdateTensorResponse(rm);
      response->Resume();
    } else if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
      RdmaTensorRequest* request = rc->GetTensorRequest(rm.request_index_);
      request->RecvErrorStatus(rm.status_);
    }
  } else if (wc_.opcode == IBV_WC_RDMA_WRITE) {
    RdmaWriteID* wr_id = reinterpret_cast<RdmaWriteID*>(wc_.wr_id);
    RDMA_LOG(2) << "Write complete of type " << wr_id->write_type;
    switch (wr_id->write_type) {
      case RDMA_WRITE_ID_ACK:
        break;
      case RDMA_WRITE_ID_MESSAGE: {
        RdmaMessageBuffer* rb =
            reinterpret_cast<RdmaMessageBuffer*>(wr_id->write_context);
        // TODO(wuyongyu02): (local buffer idle)
        rb->SetBufferStatus(local, idle);
        rb->SendNextItem();
        break;
      }
      case RDMA_WRITE_ID_SEND_DEIVER_WRITE: {
        DriverEntry* entry =
            reinterpret_cast<DriverEntry*>(wr_id->write_context);
        RDMA_LOG(1)<< "succeed send FreeEntry uid:" << entry->uinque_id_;
        break;
      }
      case RDMA_WRITE_ID_TENSOR_WRITE: {
        RdmaTensorResponse* response =
            reinterpret_cast<RdmaTensorResponse*>(wr_id->write_context);
        response->Destroy();
      }
    }
    if (wr_id->write_type != RDMA_WRITE_ID_SEND_DEIVER_WRITE) {
      delete wr_id;
    }
  }
}

void RdmaAdapter::Pool_Process_CQ(int cq_num) {
  LOG(INFO) << "Pool_Process_CQ:" << cq_num;
  auto cq = cq_vec_[cq_num];
  auto event_channel =  event_channel_vec_[cq_num];
  auto wc = wc_vec_[cq_num];
  while (true) {
    ibv_cq* cq_tmp;
    void* cq_context;
    CHECK(!ibv_get_cq_event(event_channel, &cq_tmp, &cq_context));
    CHECK(cq_tmp == cq);
    ibv_ack_cq_events(cq_tmp, 1);
    CHECK(!ibv_req_notify_cq(cq, 0));

    int ne =
        ibv_poll_cq(cq, MAX_CONCURRENT_WRITES * 2, static_cast<ibv_wc*>(wc));
    CHECK_GE(ne, 0);

    for (int i = 0; i < ne; ++i) {
      auto c = std::bind(&RdmaAdapter::Process_WR, this, wc[i], cq_num);
      pool_->Schedule(std::move(c));
      // worker_env_->compute_pool->Schedule(std::move(c));
    }
  }
}

int RdmaChannel::PingPostRecv() {
  struct ibv_recv_wr wr, *bad_wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &ping_sge_list_;
  wr.num_sge = 1;
  wr.wr_id = kPingRecvWrid;

  return ibv_post_recv(qp_, &wr, &bad_wr);
}

int RdmaChannel::PingPostSend() {
  struct ibv_send_wr wr, *bad_wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)this;
  wr.sg_list = &ping_sge_list_;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  return ibv_post_send(qp_, &wr, &bad_wr);
}

RdmaChannel::RdmaChannel(const RdmaAdapter* adapter, const string local_name,
    const string remote_name, GrpcChannelCache* channel_cache, ibv_cq* cq)
    : adapter_(adapter),
      local_name_(local_name),
      remote_name_(remote_name),
      request_serial_(0),
      could_send_driver_(false),
      channel_cache_(channel_cache),
      pd_(adapter->pd_) {

  rdma_memory_mgr_ = new RdmaMemoryMgr(adapter->pd_);
  alloc_visitors_.emplace_back([&](void* ptr, int numa_node,
                                           size_t num_bytes) {
    LOG(INFO) << "RdmaChannel RdmaMgr alloc_visitor";
    rdma_memory_mgr_->InsertMemoryRegion(
        ptr, num_bytes, strings::StrCat("CPU:", numa_node));
  });
  free_visitors_.emplace_back([&](void* ptr, int numa_node,
                                          size_t num_bytes) {
    rdma_memory_mgr_->EvictMemoryRegion(ptr, num_bytes);
  });

  rdma_mem_allocator_ = new BFCRdmaAllocator(alloc_visitors_, free_visitors_);

  struct ibv_sge list;

  mr_ = ibv_reg_mr(adapter_->pd_, ping_buff_, kPingBuffSize,
                   IBV_ACCESS_LOCAL_WRITE);
  CHECK(mr_) << "Failed to register memory region";

  memset(&list, 0, sizeof(list));
  list.addr = (uintptr_t)ping_buff_;
  list.length = kPingBuffSize;
  list.lkey = mr_->lkey;

  ping_sge_list_ = list;
  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_init_attr));
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr = adapter_->params_.queue_depth;
    attr.cap.max_recv_wr = adapter_->params_.queue_depth;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;
    // attr.qp_type = IBV_QPT_UC;

    qp_ = ibv_create_qp(adapter_->pd_, &attr);
    CHECK(qp_) << "Failed to create queue pair";
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = adapter_->params_.pkey_index;
    attr.port_num = adapter_->params_.port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    // https://man7.org/linux/man-pages/man3/ibv_modify_qp.3.html
    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "Failed to set QP to INIT";
  }

  // Local address
  {
    struct ibv_port_attr attr;
    CHECK(
        !ibv_query_port(adapter_->context_, adapter_->params_.port_num, &attr))
        << "Query port";
    self_.lid = attr.lid;
    self_.qpn = qp_->qp_num;
    self_.psn = static_cast<uint32_t>(random::New64()) & 0xffffff;
    union ibv_gid gid;
    CHECK(!ibv_query_gid(adapter_->context_, adapter_->params_.port_num,
                         adapter_->params_.sgid_index, &gid))
        << "Query gid";
    self_.snp = gid.global.subnet_prefix;
    self_.iid = gid.global.interface_id;
  }

  // create message and ack buffers, then initialize the tables.
  {
    const string buffer_names[] = {"tx_message_buffer", "rx_message_buffer"};
    message_buffers_.reserve(kNumMessageBuffers);

    // add other buffers
    for (int i = 0; i < kNumMessageBuffers; i = i + 2) {
      int pair_index = i/2+1;
      std::stringstream ss;
      ss << pair_index;
      auto* tx_buffer1 = new RdmaMessageBuffer(this,
          "tx_message_buffer_" + ss.str());
      tx_buffer1->pair_index_ = pair_index;
      auto* rx_buffer2 = new RdmaMessageBuffer(this,
          "rx_message_buffer_" + ss.str());
      rx_buffer2->pair_index_ = pair_index;
      message_buffers_.push_back(tx_buffer1);
      message_buffers_.push_back(rx_buffer2);
      // create buffer and bind to MR
      // tx_buffer1->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
      // rx_buffer2->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
      // NOTE(wuyongyu02): use chunk to alloc MR
      void* p1;
      void* p2;
      ibv_mr* mr1;
      ibv_mr* mr2;
      adapter_->recv_chunk_->Alloc(ib_allocate_size(RdmaMessage::kRdmaMessageBufferSize * 2), &p1, &mr1);
      CHECK(p1 != nullptr) << " p1 is nullptr";
      tx_buffer1->ChunkCreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize, p1, mr1);
      adapter_->recv_chunk_->Alloc(ib_allocate_size(RdmaMessage::kRdmaMessageBufferSize * 2), &p2, &mr2);
      CHECK(p1 != nullptr) << " p2 is nullptr";
      rx_buffer2->ChunkCreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize, p2, mr2);
    }
  }
  CHECK(PingPostRecv() == 0) << "Couldn't post receive from " << remote_name_
                             << " with error " << std::strerror(errno);

  channel_record_ = std::make_shared<ChannelRecordTensorMetaData>(this);
  rdma_send_driver_mgr_ = std::make_shared<RdmaSendDriverMgr>(this);
  local_driver_buffer_mgr_ = std::make_shared<LocalDriverBufferMgr>(this);
}

RdmaChannel::~RdmaChannel() {
  ibv_dereg_mr(mr_);
  CHECK(!ibv_destroy_qp(qp_)) << "Failed to destroy QP";
  // delete tx_message_buffer_;
  // delete rx_message_buffer_;
}

void RdmaChannel::SetRemoteAddress(const RdmaAddress& ra, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (!remote_set_)) {
    remote_.lid = ra.lid;
    remote_.qpn = ra.qpn;
    remote_.psn = ra.psn;
    remote_.snp = ra.snp;
    remote_.iid = ra.iid;
    remote_set_ = true;
  } else {
    CHECK(remote_.lid == ra.lid);
    CHECK(remote_.qpn == ra.qpn);
    CHECK(remote_.psn == ra.psn);
    CHECK(remote_.snp == ra.snp);
    CHECK(remote_.iid == ra.iid);
  }
}

// Adding tokens to the completion queue
// Tokens are needed to process future messages.
void RdmaChannel::Recv() {
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)this;
  struct ibv_recv_wr* bad_wr;
  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed to post recv";
}

RdmaTensorRequest* RdmaChannel::InsertTensorRequest(
    const string& key, int64 step_id, Device* dst_dev,
    const Rendezvous::Args recv_args,
    const RdmaTensorRequest::RecvDoneCallback& done) {
  mutex_lock lock{ct_mu_};
  uint32_t request_index = request_serial_++;

  // > RDMA_IMM_MIN_SENDMGR_BASE  for SendMgr
  if (request_serial_ >= RDMA_IMM_MIN_SENDMGR_BASE) {
    request_serial_ = 0;
  }

  RdmaTensorRequest request(request_index, key, step_id, this, dst_dev,
                            recv_args, done);
  auto it = request_table_.emplace(request_index, request);
  return &it.first->second;
}

void RdmaChannel::RemoveTensorRequest(uint32_t request_index) {
  mutex_lock lock{ct_mu_};
  RDMA_LOG(1) << "RemoveTensorRequest:" << request_index;
            //<< " parsed_key:" << key_;
  const auto& it = request_table_.find(request_index);
  if (it != request_table_.end()) {
    request_table_.erase(request_index);
  }
}

RdmaTensorRequest* RdmaChannel::GetTensorRequest(uint32_t request_index) {
  mutex_lock lock{ct_mu_};
  RequestTable::iterator iter = request_table_.find(request_index);
  // CHECK(iter != request_table_.end())
  //    << " RdmaChannel is already been delete.";
  if (iter == request_table_.end()) {
    return nullptr;
  }
  return &iter->second;
}

RdmaTensorRequest* RdmaChannel::GetTensorRequestForCat(uint32_t request_index) {
  mutex_lock lock{ct_mu_};
  RequestTable::iterator iter = request_table_.find(request_index);
  if (iter != request_table_.end()) {
     return &iter->second;
  }
  return nullptr;
}

void RdmaChannel::Connect() {
  {
    mutex_lock lock{mu_};
    CHECK(remote_set_) << "remote channel is not set";
  }
  Connect(remote_);
}

// Setup channel to a remote node
// Args:
//   remoteAddr: the rdma address of a remote channel.
// Returns:
//   None
void RdmaChannel::Connect(const RdmaAddress& remoteAddr) {
  mutex_lock lock{mu_};
  if (!connected_) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    // This assumes both QP's ports are configured with the same MTU
    attr.path_mtu = adapter_->params_.mtu;
    attr.dest_qp_num = remoteAddr.qpn;
    attr.rq_psn = remoteAddr.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = remoteAddr.snp;
    attr.ah_attr.grh.dgid.global.interface_id = remoteAddr.iid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.dlid = remoteAddr.lid;
    attr.ah_attr.sl = adapter_->params_.sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = adapter_->params_.port_num;
    attr.ah_attr.grh.sgid_index = adapter_->params_.sgid_index;
    attr.ah_attr.grh.traffic_class = adapter_->params_.traffic_class;

    int r;
    CHECK(!(r = ibv_modify_qp(qp_, &attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC |
                                  IBV_QP_MIN_RNR_TIMER)))
        << "QP to Ready to Receive " << r;

    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = self_.psn;
    attr.timeout = adapter_->params_.timeout;
    attr.retry_cnt = adapter_->params_.retry_cnt;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 1;

    CHECK(!(r = ibv_modify_qp(qp_, &attr,
                              IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                  IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC)))
        << "QP to Ready to Send " << r;

    connected_ = true;
  } else {
    RDMA_LOG(2) << "channel already connected";
  }
}

RdmaSendDriverMgr::RdmaSendDriverMgr(RdmaChannel* channel) {
  channel_ = channel;
  driver_mgr_is_ok_ = false;
}

size_t RdmaSendDriverMgr::InitLocalDriverEntry() {
  // LOG(INFO) << "InitLocalDriverEntry begin...";
  const auto& tensors_meta_data =
      channel_->channel_record_->GetChannelTensorsMetaData();
  const auto& global_tensors_meta_data =
      RecordTensorMetaData::Singleton().GetGlobalTensorsMetaData();
  // LOG(INFO) << "To Remote name:" << channel_->remote_name_
  //           << "Channel_Record_Size:" << tensors_meta_data.size();
  const auto& tensors_uidkeys =
      channel_->channel_record_->GetChannelTensorsUidParsedkey();

  CHECK(tensors_meta_data.size() == tensors_uidkeys.size())
      << "tensors_meta_data size:" << tensors_meta_data.size()
      << " tensors_uidkeys size:" << tensors_uidkeys.size();

  LOG(INFO) << "InitLocalDriverEntry channel Metadata key begin "
            << "create dirven-entry:"
            << tensors_meta_data.size();
  std::set<string> regrex_edge_keys;
  for (auto& it : tensors_meta_data) {
    const auto& meta_data = it.second;
    const uint32& uid = meta_data.uid_;
    void* addr;
    ibv_mr *mr;
    // allocate memory and region
    int find_allocate_bytes = 0;
    //NOTE(wuyongyu02) alloc recv-tensor memory
    if (!channel_->FindLocalMr(it.first, &addr, &mr, &find_allocate_bytes)) {
      LOG(INFO) << it.first << "not not find..";
      find_allocate_bytes = 0;
    } else {
      LOG(INFO) << it.first << "find.. bytes:" << find_allocate_bytes;
    }
    int need_bytes =  VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() *
                      ChannelRecordTensorMetaData::GetTensorBytes(meta_data) +
                      DriverPrefixMessage::kPrefixMessageTotalBytes;

    if (find_allocate_bytes < need_bytes) {
      LOG(INFO) << it.first << "reallocate find.. need:"
                << need_bytes << " " << find_allocate_bytes;
      channel_->channel_record_->AllocateMemoryAndRegion(it.first, meta_data,
          channel_->adapter_->pd_, &addr, &mr, &find_allocate_bytes);
    }
    driver_entries_[it.first] = std::make_shared<DriverEntry>(
        uid, it.first, addr, mr, find_allocate_bytes);
    driver_entries_[it.first]->meta_changed_ = meta_data.meta_changed_;
  }

  LOG(INFO) << "InitLocalDriverEntry channel Metadata key:"
            << tensors_meta_data.size()
            << " driver_entries size:"
            << driver_entries_.size()
            << " global_tensors_meta_data size:"
            << global_tensors_meta_data.size();
  // Notify local driven-entires entry through Rpc
  // Notify by Rpc
  NotifyRemoteDriverEntry();
  return driver_entries_.size();
}

// server service Update
void RdmaSendDriverMgr::RpcUpdateDriverEntries(const DriverMessageResp& resp) {
  CHECK(channel_->remote_name_ == resp.host_name())
      << "channel_->remote_name_:" << channel_->remote_name_
      << " resp.host_name:" << resp.host_name();
  size_t driver_mgr_is_ok = 0;
  for (const auto& it : resp.item()) {
    const auto& parsed_key = it.parsed_key();
    const auto& entry = driver_entries_.find(parsed_key);
    if (entry == driver_entries_.end()) {
      LOG(FATAL) << "RDMA parsed key "
                 << parsed_key
                << " is not find in driver_entries_";
      for (auto& k : driver_entries_) {
        LOG(INFO) << "kkkk:" << k.first;
      }
    }
    auto& entry_ptr = driver_entries_[parsed_key];
    if (it.status() == DriverMessageItem::RPC_0 &&
        entry_ptr->dri_status_ == RPC_0) {
      entry_ptr->dri_status_ == RPC_1;
    } else if (it.status() == DriverMessageItem::RPC_1 &&
        entry_ptr->dri_status_ == RPC_1) {
      entry_ptr->dri_status_ == DATA_NOT_READY;
      driver_mgr_is_ok++;
    } else {
      LOG(ERROR) << "RDMA RdmaSendDriverMgr::DriverEntries"
                 << " local_name:" << channel_->local_name_
                 << " remote_name:" << channel_->remote_name_
                 << " key:" << parsed_key
                 << " entry.dri_status_:" << entry_ptr->dri_status_
                 << " it.status:" << it.status();
    }
  }
  // When all entries is ok, so set driver_mgr status to 'ok'
  if (driver_mgr_is_ok == driver_entries_.size()) {
    driver_mgr_is_ok_.store(true);
    // LOG(INFO) << "[Succeed] "
    //           << channel_->remote_name_
    //           << " driver_mgr_ptr RpcSend Entries is ok!";
  }
}

bool RdmaSendDriverMgr::RpcReqResp(GrpcVerbsClient* client,
    const DriverMessageReq& req) {
  // synchronous call
  const auto& remote_name = channel_->remote_name_;
  DriverMessageResp resp;
  Status s;
  int attempts = 0;
  static const int max_num_attempts = 5;
  do {
    s = client->ReqDriverMessage(&req, &resp);
    // save obtained remote addresses
    // connect to the remote channel
    if (s.ok()) {
      RpcUpdateDriverEntries(resp);
    } else {
      LOG(ERROR) << "ReqDriverMessage Connecting to " << remote_name << ": Got "
                  << s.error_message() << ". Retrying (" << (attempts + 1)
                  << "/" << max_num_attempts << ")...";
      if (++attempts == max_num_attempts) {
        return false;
      }
      channel_->adapter_->worker_env_->env->SleepForMicroseconds(2000000);
    }
  } while (!s.ok());
  return true;
}

// Notify by Rpc
void RdmaSendDriverMgr::NotifyRemoteDriverEntry() {
  const auto& remote_name = channel_->remote_name_;
  const auto& local_name = channel_->local_name_;
  RDMA_LOG(1) << "NotifyRemoteDriverEntry local_worker_name:" << local_name
            << " remote_name:" << remote_name
            << " driver_entries_ size:" << driver_entries_.size();

  auto* cache = channel_->channel_cache_;
  // get the channel cache
  SharedGrpcChannelPtr client_channel =
      channel_->channel_cache_->FindWorkerChannel(remote_name);
  CHECK(client_channel != nullptr) << "target:"
                                   << remote_name
                                   << " client_channel is null!";
  GrpcVerbsClient* client = new GrpcVerbsClient(client_channel);
  CHECK(client != nullptr) << "No worker known as " << remote_name;

  DriverMessageReq req;
  req.set_host_name(local_name);
  for (auto& it : driver_entries_) {
    auto* item = req.add_item();
    auto driver_entry_ptr = it.second;
    item->set_unique_id(driver_entry_ptr->uinque_id_);
    item->set_parsed_key(it.first);
    item->set_remote_addr(driver_entry_ptr->addr_);
    item->set_rkey(driver_entry_ptr->lkey_);
    item->set_allocate_bytes(driver_entry_ptr->allocate_size_);
    item->set_meta_changed(driver_entry_ptr->meta_changed_);
    item->set_status(DriverMessageItem::RPC_0);
    // Remember to update driver_entries_ Status
    it.second->dri_status_ = RPC_0;
  }
  if (RpcReqResp(client, req)) {
    DriverMessageReq req_rpc2;
    req_rpc2.set_host_name(local_name);
    for (auto& it : driver_entries_) {
      auto* item = req_rpc2.add_item();
      auto driver_entry_ptr = it.second;
      item->set_unique_id(driver_entry_ptr->uinque_id_);
      item->set_parsed_key(it.first);
      item->set_remote_addr(driver_entry_ptr->addr_);
      item->set_rkey(driver_entry_ptr->lkey_);
      item->set_allocate_bytes(driver_entry_ptr->allocate_size_);
      item->set_meta_changed(driver_entry_ptr->meta_changed_);
      item->set_status(DriverMessageItem::RPC_1);
      // Remember to update driver_entries_ Status
      it.second->dri_status_ = RPC_1;
    }
    if (!RpcReqResp(client, req_rpc2)) {
      LOG(ERROR) << "ReqDriverMessage RpcReqResp2 remote node "
               << remote_name << " FAILED";
    }
  } else {
    LOG(ERROR) << "ReqDriverMessage RpcReqResp remote node "
               << remote_name << " FAILED";
  }
  RDMA_LOG(0) << "ReqDriverMessage Connected to remote node " << remote_name;
  delete client;
}

void RdmaSendDriverMgr::RpcUpdateRemoteDriverEntry(
    const DriverMessageReq* request, DriverMessageResp* response) {
  // setting up response
  response->set_host_name(channel_->local_name_);
  int recv_driver_mgr_entry_ok_nums = 0;
  for (const auto& req_item : request->item()) {
    DriverMessageItem* resp_item = response->add_item();
    string parsed_key = req_item.parsed_key();
    resp_item->set_parsed_key(parsed_key);
    const auto& it = recv_entries_.find(parsed_key);
    DriverMessageItem::DriverStatus status = req_item.status();
    if (it == recv_entries_.end() && status == DriverMessageItem::RPC_0) {
      recv_entries_[parsed_key] = std::make_shared<DriverEntry>();
      recv_entries_[parsed_key]->uinque_id_ = req_item.unique_id();
      recv_entries_[parsed_key]->addr_ =  req_item.remote_addr();
      recv_entries_[parsed_key]->lkey_ = req_item.rkey();
      recv_entries_[parsed_key]->allocate_size_ = req_item.allocate_bytes();
      recv_entries_[parsed_key]->meta_changed_ = req_item.meta_changed();
      recv_entries_[parsed_key]->parsed_key_ = parsed_key;
      // update recv entries
      recv_entries_[parsed_key]->dri_status_ = RPC_0;
      // response status
      resp_item->set_status(DriverMessageItem::RPC_0);
      RDMA_LOG(1) << "RpcUpdateRemoteDriverEntry parsed_key :"
                << parsed_key
                << " recv dir_status: RPC_0 "
                << " update dir_status RPC_1 : "
                << recv_entries_[parsed_key]->dri_status_;
    } else if (it->second->dri_status_ == RPC_0 &&
               status == DriverMessageItem::RPC_1) {
      // response status
      resp_item->set_status(DriverMessageItem::RPC_1);
      // update recv entries
      recv_entries_[parsed_key]->dri_status_ = DATA_NOT_READY;
      recv_driver_mgr_entry_ok_nums += 1;
      RDMA_LOG(1) << "RpcUpdateRemoteDriverEntry parsed_key :"
                << parsed_key
                << " recv dir_status: RPC_1 "
                << " update dir_status DATA_NOT_READY : "
                << recv_entries_[parsed_key]->dri_status_;
    } else {
      LOG(ERROR) << "UpdateRemoteDriverEntry:"
                  << "local_name:"
                  << channel_->local_name_
                  << " revc from remote:"
                  << request->host_name()
                  << " parsed_key:"
                  << parsed_key
                  << " recv_entries dri_status is not `RPC_1` "
                  << " status is :"
                  << status
                  << " dri_status is "
                  << recv_entries_[parsed_key]->dri_status_;
    }
  }
  RDMA_LOG(1) << "RdmaSendDriverMgr::RpcUpdateRemoteDriverEntry end...."
            << " localname:" << channel_->local_name_
            << " remotename:" << channel_->remote_name_
            << " recv_entries_ size:" << recv_entries_.size();
  // driver_mgr_ptr is ok and can send tensor to other client.
  if (recv_driver_mgr_entry_ok_nums == recv_entries_.size()) {
    // allocate string RDMA
    // NOTE(wuyongyu02)
    // Allocate StringMessage change to FindOrCreateMemeoryRegion
    // AllocateRecvEntriesStringMemoryAndRegion();
    LOG(INFO) << "[Succeed] "
              << request->host_name()
              << " driver_mgr_ptr RecvEntries is ok!";
  }
}

void RdmaSendDriverMgr::AllocateRecvEntriesStringMemoryAndRegion() {
  for (auto& k : recv_entries_) {
    void* addr;
    ibv_mr *mr;
    // allocate memory and region
    int allocate_bytes = 0;
    channel_->channel_record_->AllocateSendStringMemoryAndRegion(
        channel_->adapter_->pd_, &addr, &mr, &allocate_bytes);
    k.second->send_mem_mr_ = std::make_shared<RemoteBytesAddrMemoryRegion>(
        addr, mr, allocate_bytes);
    RDMA_LOG(1) << "AllocateRecvEntriesStringMemoryAndRegion:"
              << k.first
              << " allocate_bytes:"
              << allocate_bytes;
  }
}

std::shared_ptr<DriverEntry> RdmaSendDriverMgr::GetRecvEntry(
    const std::string& parsed_key, bool* has_data) {
  const auto& it = recv_entries_.find(parsed_key);
  if (it == recv_entries_.end()) {
    for (auto& find : recv_entries_) {
      if (absl::StrContains(parsed_key, find.first)) {
        return find.second;
      }
    }
    // LOG(FATAL) << parsed_key << " is not find in recv_entries_.";
    return nullptr;
  }
  *has_data = recv_entries_[parsed_key]->dri_status_ == DATA_READY;
  // LOG(INFO) << "parsed_key:" << parsed_key
  //           << " status:" << recv_entries_[parsed_key]->dri_status_
  //           << " has_data:" << *has_data;
  return recv_entries_[parsed_key];
}

std::shared_ptr<DriverEntry> RdmaSendDriverMgr::GetDriverEntry(
    const std::string& parsed_key, bool* has_data) {
  const auto& it = driver_entries_.find(parsed_key);
  if (it == driver_entries_.end()) {
    for (auto& find : driver_entries_) {
      if (absl::StrContains(parsed_key, find.first)) {
        return find.second;
      }
    }
    LOG(FATAL) << parsed_key << " is not find in driver_entries_.";
  }
  *has_data = driver_entries_[parsed_key]->dri_status_ == DATA_READY;
  return driver_entries_[parsed_key];
}

DriverEntry::DriverEntry() {
  dri_status_.store(DRIVER_INIT);
}

DriverEntry::DriverEntry(const uint32& uid,
              const std::string& parsedkey,
              void* addr,
              ibv_mr* mr,
              int allocate_size) {
  addr_ = (uint64_t) addr;
  mem_mr_ =
      std::make_shared<RemoteBytesAddrMemoryRegion>(addr, mr, allocate_size);
  lkey_ = mr->lkey;
  uinque_id_ = uid;
  parsed_key_ = parsedkey;
  dri_status_.store(DRIVER_INIT);
  allocate_size_ = allocate_size;
}

string ChannelRecordTensorMetaData::RegexEdgeName(const string & str) {
  std::string regex_str(".*edge_\\d*(_.*)(_\\d*)?;0:0");
  std::regex pattern(regex_str, std::regex::icase);
  std::smatch result;
  if (std::regex_match(str, result, pattern)) {
    return std::string(result[1]);
  } else {
    LOG(ERROR) << "RegexEdgeName key:" << str << " is not matchaed. pattern:"
                << regex_str;
  }
  return str;
}

void ChannelRecordTensorMetaData::InitMetaDataFromEnv() {
  // Init Channel
  mutex_lock l(channel_tensor_meta_data_mu_);
  const string& name = channel_->local_name_;
  if (absl::StrContains(name, "worker") ||
      absl::StrContains(name, "localhost")) {
    const string meta_str = GetWorkerMetas();
    StringPiece s(meta_str);
    while (!s.empty()) {
      StringPiece result = ConsumeNextPart(&s, '|');
      if (!result.empty()) {
        StringPiece meta_name_view = ConsumeNextPart(&result, '#');
        if (!meta_name_view.empty()) {
          auto meta_name = string(meta_name_view);
          std::stringstream ss(string(result).c_str());
          int meta_size = 0;
          ss >> meta_size;
          CHECK(meta_size > 0)
              << " meta_name" << meta_name << " size:" << meta_size;
          auto find = channel_tensors_meta_data_.find(meta_name);
          if (find == channel_tensors_meta_data_.end()) {
            auto it = channel_tensors_meta_data_.emplace(meta_name,
                                                         TensorMetaData());
            channel_tensors_uid_parsed_key_.emplace(uid_, meta_name);
            auto& meta = channel_tensors_meta_data_[meta_name];
            meta.uid_ = uid_;
            if (it.second) {
              uid_++;
            }
            meta.data_type_ = DT_INT64;
            meta.tensor_shape_ = {};
            meta.proto_size_ = 0;
            meta.is_dead_ = false;
          }
        }
      }
    }
  }
}

ChannelRecordTensorMetaData::ChannelRecordTensorMetaData(RdmaChannel* channel) {
  channel_ = channel;
  InitMetaDataFromEnv();
}

uint32 ChannelRecordTensorMetaData::GetEnumSize(const DataType& date_type) {
  switch (date_type) {
    case DT_FLOAT:
      return 4;
      break;
    case DT_DOUBLE:
      return 8;
      break;
    case DT_INT32:
      return 4;
      break;
    case DT_UINT32:
      return 4;
      break;
    case DT_UINT16:
      return 2;
      break;
    case DT_INT8:
      return 1;
      break;
    case DT_UINT8:
      return 1;
      break;
    case DT_INT16:
      return 2;
      break;
    case DT_INT64:
      return 8;
      break;
    case DT_UINT64:
      return 8;
      break;
    case DT_BOOL:
      return 1;
      break;
    default:
      return 4;
      break;
  }
}

void ChannelRecordTensorMetaData::AllocateSendStringMemoryAndRegion(ibv_pd* pd,
                                        void** addr,
                                        ibv_mr** mr,
                                        int* addr_size,
                                        Allocator* alloc_attr) {
  // allocate prefix DriverPrefixMessage
  auto total_bytes = DriverPrefixMessage::kPrefixMessageTotalBytes;
  RDMA_LOG(1) << "AllocateSendStringMemoryAndRegion total bytes:"
              << total_bytes;
  *addr = malloc(total_bytes);
  CHECK(addr != nullptr)
      << "AllocateSendStringMemoryAndRegion addr malloc faild!";
  *mr = ibv_reg_mr(pd, *addr, total_bytes,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  *addr_size = total_bytes;
}

int ChannelRecordTensorMetaData::GetTensorBytes(const TensorMetaData& m) {
  int total_bytes = 0;
  if (DataTypeCanUseMemcpy(m.data_type_)) {
    int m1 = m.tensor_shape_.num_elements();
    total_bytes = m1 * GetEnumSize(m.data_type_);
  } else {
    total_bytes = m.proto_size_;
  }
  return total_bytes;
}

void ChannelRecordTensorMetaData::AllocateMemoryAndRegion(
                                  const string& key,
                                  const TensorMetaData& m,
                                  ibv_pd* pd,
                                  void** addr,
                                  ibv_mr** mr,
                                  int* addr_size,
                                  Allocator* alloc_attr) const {
  int total_bytes = GetTensorBytes(m);
  total_bytes =
      VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * total_bytes;
  // allocate prefix DriverPrefixMessage
  total_bytes += DriverPrefixMessage::kPrefixMessageTotalBytes;
  RDMA_LOG(1) << "AllocateMemoryAndRegion key:"
              << key
              << " total bytes:" << total_bytes;
  *addr = malloc(total_bytes);
  CHECK(addr != nullptr) << "AllocateMemoryAndRegion addr malloc faild!";
  *mr = ibv_reg_mr(pd, *addr, total_bytes,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  *addr_size = total_bytes;
}


void ChannelRecordTensorMetaData::Record(const std::string& tensor_name,
                                         const TensorMetaData& m) {
  // send-driver stop record
  if (channel_->could_send_driver_) {
    return;
  }
  // LOG(INFO) << "ChannelRecordTensorMetaData::Record " << is_stable_;
  mutex_lock l(channel_tensor_meta_data_mu_);
  auto find = channel_tensors_meta_data_.find(tensor_name);
  if (find == channel_tensors_meta_data_.end()) {
    // LOG(INFO) << "Channel Record Tensorname:" << tensor_name;
    auto it = channel_tensors_meta_data_.emplace(tensor_name, TensorMetaData());
    channel_tensors_uid_parsed_key_.emplace(uid_, tensor_name);
    auto& meta = channel_tensors_meta_data_[tensor_name];
    meta.uid_ = uid_;
    if (it.second) {
      uid_++;
    }
    meta.data_type_ = m.data_type_;
    meta.tensor_shape_ = m.tensor_shape_;
    meta.proto_size_ = m.proto_size_;
    meta.is_dead_ = m.is_dead_;
  } else {
    auto& meta = find->second;
    bool can_memcpy = DataTypeCanUseMemcpy(m.data_type_);
    if (can_memcpy) {
      int m1 = 1;
      int m2 = 1;
      for (int d = 0; d < m.tensor_shape_.dims(); d++) {
        m1 *= m.tensor_shape_.dim_size(d);
        m2 *= meta.tensor_shape_.dim_size(d);
      }
      if (m1 > m2) {
        meta.data_type_ = m.data_type_;
        meta.tensor_shape_ = m.tensor_shape_;
        meta.proto_size_ = m.proto_size_;
        meta.is_dead_ = m.is_dead_;
      }
      if (m1 != m2) {
        // LOG(INFO) << "Tensorname:" << tensor_name << " meta_changed.";
        meta.meta_changed_ = true;
      }
    }
    if ((!can_memcpy && meta.proto_size_ > m.proto_size_)) {
      meta.data_type_ = m.data_type_;
      meta.tensor_shape_ = m.tensor_shape_;
      meta.proto_size_ = 10 * m.proto_size_;
      meta.is_dead_ = m.is_dead_;
    }
    if (!can_memcpy && meta.proto_size_ != m.proto_size_) {
      // LOG(INFO) << "Tensorname:" << tensor_name << " _meta_changed.";
      meta.meta_changed_ = true;
    }
  }
}

StringPiece ChannelRecordTensorMetaData::ConsumeNextPart(StringPiece* s,
    char delim) {
  for (size_t offset = 0; offset < s->size(); offset++) {
    if ((*s)[offset] == delim) {
      StringPiece result(s->data(), offset);
      s->remove_prefix(offset + 1);  // +1: remove delim, as well
      return result;
    }
  }
  // No delimiter found: return rest of string
  StringPiece result(s->data(), s->size());
  s->remove_prefix(s->size());
  return result;
}

string RecordTensorMetaData::DebugString() const {
  std::vector<string> lc;
  for (auto& it : global_tensors_meta_data_) {
    std::vector<string> ds;
    ds.emplace_back(string(it.first));
    // dtype
    ds.emplace_back(std::to_string(it.second.data_type_));
    // num elements
    auto num_elements = it.second.tensor_shape_.num_elements();
    ds.emplace_back(std::to_string(num_elements));
    auto total_bytes = num_elements * GetEnumSize(it.second.data_type_);
    ds.emplace_back(std::to_string(total_bytes));
    lc.push_back(absl::StrJoin(lc, ","));
  }
    return absl::StrJoin(lc, "\n");
}

void RecordTensorMetaData::WriteOutput(const std::string& content) const {
  Env* env = Env::Default();
  std::string path_dir = GetMetaOutput();
  if (!env->FileExists(path_dir).ok()) {
    LOG(INFO) << "File " << path_dir << " is not exists!";
    env->CreateDir(path_dir);
    LOG(INFO) << "CreateFileDir " << path_dir << " sucess!";
  }

  std::string cfn = path_dir + "/" + local_worker_name_;
  // Write something to the temporary file.
  std::unique_ptr<WritableFile> file_to_write;
  TF_CHECK_OK(env->NewWritableFile(cfn, &file_to_write));
  TF_CHECK_OK(file_to_write->Append(content));
  TF_CHECK_OK(file_to_write->Close());
  TF_CHECK_OK(env->FileExists(cfn));
}

void RecordTensorMetaData::ReadFile(const std::string& filename,
    StringPiece* content) {
  Env* env = Env::Default();
  // Read from the temporary file and check content.
  std::unique_ptr<RandomAccessFile> file_to_read;
  TF_CHECK_OK(env->NewRandomAccessFile(filename, &file_to_read));
  // StringPiece content;
  char scratch[1024];
  CHECK_EQ(error::OUT_OF_RANGE,
          file_to_read->Read(0 /* offset */, 1024 /* n */, content, scratch)
              .code());
}

uint32 RecordTensorMetaData::GetEnumSize(const DataType& date_type) {
  switch (date_type) {
    case DT_FLOAT:
      return 4;
      break;
    case DT_DOUBLE:
      return 8;
      break;
    case DT_INT32:
      return 4;
      break;
    case DT_UINT32:
      return 4;
      break;
    case DT_UINT16:
      return 2;
      break;
    case DT_INT8:
      return 1;
      break;
    case DT_UINT8:
      return 1;
      break;
    case DT_INT16:
      return 2;
      break;
    case DT_INT64:
      return 8;
      break;
    case DT_UINT64:
      return 8;
      break;
    case DT_BOOL:
      return 1;
      break;
    default:
      return 4;
      break;
  }
}

void RecordTensorMetaData::GlobalRecord(const std::string& origin_tensor_name,
    const TensorMetaData& m, bool stop_record) {
  // send-driver status stop record
  if (stop_record) {
    return;
  }
  mutex_lock l(global_tensor_meta_data_mu_);
  auto tensor_name = ChannelRecordTensorMetaData::RegexEdgeName(
      origin_tensor_name);
  auto find = global_tensors_meta_data_.find(tensor_name);
  // LOG(INFO) << "Record Tensorname:" << tensor_name;
  if (find == global_tensors_meta_data_.end()) {
    auto it = global_tensors_meta_data_.emplace(tensor_name, TensorMetaData());
    global_tensors_uid_parsed_key_.emplace(uid_, tensor_name);
    auto& meta = global_tensors_meta_data_[tensor_name];
    meta.uid_ = uid_;
    if (it.second) {
      uid_++;
    }
    meta.data_type_ = m.data_type_;
    meta.tensor_shape_ = m.tensor_shape_;
    meta.proto_size_ = m.proto_size_;
    meta.is_dead_ = m.is_dead_;
  } else {
    auto& meta = find->second;
    bool can_memcpy = DataTypeCanUseMemcpy(m.data_type_);
    if (can_memcpy) {
      int m1 = 1;
      int m2 = 1;
      for (int d = 0; d < m.tensor_shape_.dims(); d++) {
        m1 *= m.tensor_shape_.dim_size(d);
        m2 *= meta.tensor_shape_.dim_size(d);
      }
      if (m1 > m2) {
        meta.data_type_ = m.data_type_;
        meta.tensor_shape_ = m.tensor_shape_;
        meta.proto_size_ = m.proto_size_;
        meta.is_dead_ = m.is_dead_;
      }
    }
    if ((!can_memcpy && meta.proto_size_ > m.proto_size_)) {
      meta.data_type_ = m.data_type_;
      meta.tensor_shape_ = m.tensor_shape_;
      meta.proto_size_ = 10 * m.proto_size_;
      meta.is_dead_ = m.is_dead_;
    }
  }
  int tmp_sizes = 0;
  for(const auto& k : global_tensors_meta_data_) {
    tmp_sizes += ChannelRecordTensorMetaData::GetTensorBytes(k.second);
  }
  if (tmp_sizes > total_bytes_) {
    total_bytes_ = tmp_sizes;
    LOG(INFO) << "GlobalRecord bytes:" << total_bytes_;
  }
}

RdmaMessageBuffer::RdmaMessageBuffer(RdmaChannel* channel, string name)
    : channel_(channel), name_(name) {}

RdmaMessageBuffer::~RdmaMessageBuffer() {
  CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
  FreeBuffer();
}

void RdmaMessageBuffer::FreeBuffer() {
  if ((buffer_ != nullptr) && buffer_on_host_) {
    free(buffer_);
  }
}

void RdmaMessageBuffer::ChunkCreateCPUBuffer(size_t size, void* buffer, ibv_mr* mr,
    bool lock) {
  CHECK(size > 0);
  if (lock) {
    mu_.lock();
  }
  if (local_status_ != none) {
    // delete existing buffer
  }
  size_ = size;
  buffer_ = buffer;
  self_ = mr;
  CHECK(self_) << "Failed to register memory region";
  buffer_on_host_ = true;
  local_status_ = idle;
  if (lock) {
    mu_.unlock();
  }
}

// Allocate CPU memory for the Rdma buffer
// Args:
//   size: to-be-allocated memory size
//   lock: whether or not mutex_lock the process to protect concurrency.
// Returns:
//   None
void RdmaMessageBuffer::CreateCPUBuffer(size_t size, bool lock) {
  CHECK(size > 0);
  if (lock) {
    mu_.lock();
  }
  if (local_status_ != none) {
    // delete existing buffer
    CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
    FreeBuffer();
  }
  size_ = size;
  buffer_ = malloc(size_);
  self_ = ibv_reg_mr(channel_->adapter_->pd_, buffer_, size_,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHECK(self_) << "Failed to register memory region";
  buffer_on_host_ = true;
  local_status_ = idle;
  if (lock) {
    mu_.unlock();
  }
}

// Set address of remote memory region
// Args:
//   rmr: address of remote memory region
//   override: whether override existing information
// Returns:
//   None
void RdmaMessageBuffer::SetRemoteMR(RemoteMR rmr, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (remote_status_ == none)) {
    remote_.remote_addr = rmr.remote_addr;
    remote_.rkey = rmr.rkey;
    remote_status_ = idle;
  } else {
    CHECK(remote_.remote_addr == rmr.remote_addr);
    CHECK(remote_.rkey == rmr.rkey);
  }
}

// Put a task in the buffer's job queue
void RdmaMessageBuffer::EnqueueItem(string item) {
  mutex_lock lock{mu_};
  queue_.push(item);
}

// Rdma-Write the content of the buffer
void RdmaMessageBuffer::Write(uint32_t imm_data, size_t buffer_size) {
  Write(channel_, imm_data, buffer_size, (uint64_t)buffer_, self_->lkey,
        remote_.remote_addr, remote_.rkey, RDMA_WRITE_ID_MESSAGE, this);
}

// Generalized Write method
void RdmaMessageBuffer::WriteWithPrefix(const RdmaChannel* channel,
                              uint32_t imm_data,
                              size_t buffer_size,
                              uint64_t src_addr,
                              uint32_t lkey,
                              uint64_t remote_addr,
                              uint32_t rkey,
                              RdmaWriteIDType write_type,
                              void* write_context,
                              uint64_t prefix_addr,
                              uint32_t prefix_lkey,
                              size_t prefix_size) {
  struct ibv_sge* list = new ibv_sge[2];
  list[0].addr = prefix_addr;
  list[0].length = prefix_size;
  list[0].lkey = prefix_lkey;

  list[1].addr = src_addr;
  list[1].length = buffer_size;
  list[1].lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  wr.wr_id = (uint64_t) new RdmaWriteID(write_type, write_context);
  wr.sg_list = list;
  wr.num_sge = 2;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel->qp_, &wr, &bad_wr)) << "Failed to post send";
}

// Generalized Write method
void RdmaMessageBuffer::Write(const RdmaChannel* channel, uint32_t imm_data,
                              size_t buffer_size, uint64_t src_addr,
                              uint32_t lkey, uint64_t remote_addr,
                              uint32_t rkey, RdmaWriteIDType write_type,
                              void* write_context) {
  struct ibv_sge list;
  list.addr = src_addr;
  list.length = buffer_size;
  list.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  wr.wr_id = (uint64_t) new RdmaWriteID(write_type, write_context);
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel->qp_, &wr, &bad_wr)) << "Failed to post send";
}

// Send the next ack from the buffer's job queue.
void RdmaMessageBuffer::SendAck(const RdmaChannel* channel, int pair_index) {
  Write(channel, RDMA_IMM_MAX_REQUEST_ID + pair_index, 0, 0, 0, 0, 0,
        RDMA_WRITE_ID_ACK, nullptr);
}

// Send the next message from the buffer's job queue.
void RdmaMessageBuffer::SendNextItem() {
  uint32_t imm_data = RDMA_IMM_DATA_ACK + pair_index_;
  mu_.lock();
  if (!queue_.empty() && (local_status_ == idle) && (remote_status_ == idle)) {
    local_status_ = busy;
    remote_status_ = busy;
    time_guard_ = 0;
    rm_ack_micros_ = 0;
    // LOG(ERROR) << "SendNextItem queue size:" << queue_.size();
    string message = queue_.front();
    queue_.pop();
    // local/remote_status_ won't be set back to idle
    // unitl Write() is successful
    mu_.unlock();
    memcpy(buffer_, message.data(), message.size());
    Write(imm_data, message.size());
  } else {
    mu_.unlock();
  }
}

#if GOOGLE_CUDA
static void CountCopies(const std::string& key, void* src_addr, void* dst_addr,
                        size_t tensor_bytes, bool is_gpu_to_cpu) {
#ifdef RDMA_COUNT_COPIES
  static uint64_t numGPUToCPUCopies = 0;
  static uint64_t numGPUToCPUCopiedBytes = 0;
  static uint64_t numCPUToGPUCopies = 0;
  static uint64_t numCPUToGPUCopiedBytes = 0;
  static uint64_t numTotalCopies = 0;

  if (is_gpu_to_cpu) {
    ++numGPUToCPUCopies;
    numGPUToCPUCopiedBytes += tensor_bytes;
  } else {
    ++numCPUToGPUCopies;
    numCPUToGPUCopiedBytes += tensor_bytes;
  }
  if ((++numTotalCopies % 0x400) == 0) {
    RDMA_LOG(0) << "Tensor copies:"
                << " GPU to CPU: " << numGPUToCPUCopies << " ("
                << numGPUToCPUCopiedBytes << " Bytes)"
                << " CPU to GPU: " << numCPUToGPUCopies << " ("
                << numCPUToGPUCopiedBytes << " Bytes)";
  }
  RDMA_LOG(2) << "Copying tensor " << key << " From: " << src_addr
              << " To: " << dst_addr;
#endif  // RDMA_COUNT_COPIES
}
#endif  // GOOGLE_CUDA

#ifdef RDMA_DATA_VALIDATION
static uint64_t Checksum(Device* device, const DeviceContext* device_context,
                         const Tensor& in) {
  uint64 checksum = 0;
  if (DataTypeCanUseMemcpy(in.dtype())) {
#if GOOGLE_CUDA
    if (in.TotalBytes() == 0) {
      return 0;
    }
    checksum = (device_context != nullptr)
                   ? GPUUtil::Checksum(device, device_context, in)
                   : GPUUtil::Checksum(in);
#endif  // GOOGLE_CUDA
  } else {
    string s = in.SummarizeValue(999999);
    checksum = Hash64(s.c_str(), s.size(), 0);
  }
  return checksum;
}

static void ValidateChecksum(uint64_t expected, uint64_t actual,
                             const Tensor& in, uint32_t request_index,
                             const std::string& key, const std::string& msg) {
  RDMA_LOG(2) << "Request #" << request_index << ": " << key
              << ": Checksum: " << std::hex << " Expected = 0x" << expected
              << ". Actual = 0x" << actual << ".";

  if (expected != actual) {
    // Checksum failed. There is one case where this is allowed - if the
    // tensor is an AssignAdd of the global step. Since the data-validation
    // always postpones the Tensor response in order to send a checksum message,
    // it is possible that the global-step was updated while the response was
    // still in queue.
    if ((in.TotalBytes() == 8) && (in.dtype() == DT_INT64)) {
      int64_t prev_val = *(int64_t*)DMAHelper::base(&in) - 1;
      actual = Hash64((const char*)&prev_val, 8, 0);
    }
    if (expected != actual) {
      LOG(FATAL) << "[" << msg << "]: Checksum validation failed for request #"
                 << request_index << ": " << key << std::hex << " "
                 << DataTypeString(in.dtype()) << " "
                 << in.shape().DebugString() << " (0x" << in.TotalBytes()
                 << " bytes): "
                 << " Expected 0x" << expected << ". Got 0x" << actual << ".";
    }
  }
}
#endif  // RDMA_DATA_VALIDATION

#if GOOGLE_CUDA
// Sync the 'done' operation on the GPU stream, but without all the data
// copying.
static void StreamGPUOp(Device* gpu_device, const DeviceContext* device_context,
                        StatusCallback done) {
  Tensor dummy1, dummy2;
  GPUUtil::CopyGPUTensorToCPU(gpu_device, device_context, &dummy1, &dummy2,
                              done);
}
#endif  // GOOGLE_CUDA

RdmaTensorResponse* RdmaChannel::AddTensorResponse(const RdmaMessage& rm) {
  mutex_lock lock{mu_};
  auto it = responses_table_.emplace(rm.request_index_,
      std::make_shared<RdmaTensorResponse>(this, rm));
  CHECK(it.second) << "Response with the ID " << rm.request_index_
                   << " already exists.";
  // replica request_index
  it.first->second->request_index_ = rm.request_index_;
  return it.first->second.get();
}

RdmaTensorResponse* RdmaChannel::UpdateTensorResponse(const RdmaMessage& rm) {
  mutex_lock lock{mu_};
  auto it = responses_table_.find(rm.request_index_);
  CHECK(it != responses_table_.end()) << "No response found.";
  RdmaTensorResponse* response = it->second.get();
  response->Update(rm);
  return response;
}

void RdmaChannel::RemoveTensorResponse(uint32_t request_index) {
  mutex_lock lock{mu_};
  if (responses_table_.find(request_index) != responses_table_.end())
    responses_table_.erase(request_index);
}

void RdmaTensorResponse::Start() {
  // LOG(INFO) << "RdmaTensorResponse Start...";
  Rendezvous::ParsedKey parsed;
  Status s = Rendezvous::ParseKey(rm_.name_, &parsed);
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev_);
  }
  if (!s.ok()) {
    SendErrorStatus(s, "RdmaTensorResponse::Start::PrepareRecvTensor");
    return;
  }
  recv_local_send_rdma_ = 0;
  channel_->adapter_->worker_env_->rendezvous_mgr->RecvLocalAsync(
      rm_.step_id_, parsed,
      [this](const Status& status, const Rendezvous::Args& send_args,
                     const Rendezvous::Args& recv_args, const Tensor& in,
                     bool is_dead) mutable {
          // (wuyongyu02) if the sender don't receive tensor from local
          // should't CHECK(FALSE), can send SendErrorStatus  like
          // RdmaTensorResponse::RecvHandler(...) {... SendErrorStatus(status);}
          // CHECK(status.ok()) << "RecvLocalAsync was not ok."
          //                    << "src_device : " << parsed.src_device
          //                    << "dst_device : " << parsed.dst_device
          //                    << " error message: " << status.error_message();
          if (!status.ok()) {
            // SendErrorStatus(status, "rendezvous_mgr->RecvLocalAsync::");
            return;
          }
          RecvHandler(send_args, recv_args, in, is_dead);
      });
}

void RdmaTensorResponse::Resume() { SendContent(*tensor_, *proto_, is_dead_, true); }

// Helper for RecvTensor. Validates "key" and returns the source
// device in "*src_dev".
Status RdmaTensorResponse::PrepareRecvTensor(
    const Rendezvous::ParsedKey& parsed, Device** src_dev) {
  // Figures out which device the tensor is hosted on.
  string local_name = DeviceNameUtils::LocalName(parsed.src_device);
  TF_RETURN_IF_ERROR(channel_->adapter_->worker_env_->device_mgr->LookupDevice(
      local_name, src_dev));

  // Does the device have the right incarnation number we expect?
  if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
    return errors::Aborted(
        "RecvTensor expects a different device incarnation: ",
        parsed.src_incarnation, " vs. ", (*src_dev)->attributes().incarnation(),
        ". Your worker job (\"",
        channel_->adapter_->worker_env_->session_mgr->LegacySession()
            ->worker_name,
        "\") was probably restarted. Check your "
        "worker job for the reason why it was restarted.");
  }

  return Status::OK();
}

void RdmaTensorResponse::RecvHandler(const Rendezvous::Args& send_args,
                                     const Rendezvous::Args& recv_args,
                                     const Tensor& in, bool is_dead) {
  meta_data_changed_ = TensorMetaDataChanged(in, is_dead);
#ifdef RDMA_DATA_VALIDATION
  // Always send a meta data message with the source checksum
  meta_data_changed_ = rm_.type_ == RDMA_MESSAGE_TENSOR_REQUEST;
  checksum_ = Checksum(src_dev_, send_args.device_context, in);
#endif
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  // string tensor needs to be serialized
  Tensor copy;
  TensorProto proto;
  const bool on_host = send_args.alloc_attrs.on_host();
  if (src_dev_->tensorflow_gpu_device_info() && !on_host) {
#if GOOGLE_CUDA
    DeviceContext* send_dev_context = send_args.device_context;
    CHECK(send_dev_context)
        << "send dev name: " << src_dev_->name()
        << " gpu_info: " << src_dev_->tensorflow_gpu_device_info();

    if (can_memcpy) {
      // If the tensor is located on a GDR compatible GPU, there is no need to
      // copy it. We can send directly from the source, just need to make sure
      // we are in sync with the GPU stream.
      // If the tensor's meta-data changed however, we will need to clone it,
      // so anyway we'll have to copy it from GPU to CPU first. If at some
      // point in time Clone() is changed to only save a shallow copy, we can
      // skip the copy here as well.
      if ((in.TotalBytes() > 0) && !meta_data_changed_) {
        StreamGPUOp(src_dev_, send_dev_context,
                    [this, in, proto, is_dead](const Status& s) {
                      Send(in, proto, is_dead, s);
                    });
        return;
      }

      // The tensor must be copied from GPU to CPU, because either:
      // 1. The tensor is located on a non GDR compatible GPU.
      // 2. The tensor's meta-data has changed.
      Allocator* alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
      copy = Tensor(alloc, in.dtype(), in.shape());
      CountCopies(rm_.name_, (void*)DMAHelper::base(&in),
                  (void*)DMAHelper::base(&copy), in.TotalBytes(), true);
      GPUUtil::CopyGPUTensorToCPU(
          src_dev_, send_dev_context, &in, &copy,
          [this, copy, proto, is_dead](const Status& s) {
            Send(copy, proto, is_dead, s);
          });
    } else {
      GPUUtil::SetProtoFromGPU(
          in, src_dev_, send_args.device_context, &proto, is_dead,
          [this, in, proto, is_dead](const Status& s) mutable {
            Send(in, proto, is_dead, s);
          });
    }
#else
    SendErrorStatus(errors::Internal("No GPU device in process"),
                    "No GPU device in process");
#endif  // GOOGLE_CUDA
  } else {
    // tensor is in CPU memory.
    if (!can_memcpy) {
      in.AsProtoTensorContent(&proto);
    }
    Send(in, proto, is_dead, Status::OK());
  }
}

void RdmaTensorResponse::Send(const Tensor& in, const TensorProto& proto,
                              bool is_dead, const Status& status) {
  if (!status.ok()) {
    SendErrorStatus(status, "RdmaTensorResponse::Send::!status.ok");
    return;
  }
  SendBck(in, proto, is_dead, status);
}

void RdmaChannel::SendDriverData(const Tensor& in,
                                 bool is_dead,
                                 const std::string& name) {
  bool has_data = false;
  std::shared_ptr<DriverEntry> entry =
      rdma_send_driver_mgr_->GetRecvEntry(name, &has_data);

  CHECK(entry.get() != nullptr) << "Channel SendDriverData to "
                                << name
                                << " is_dead:"
                                << is_dead
                                << " dtype:"
                                << DataTypeString(in.dtype())
                                << " shape:"
                                << in.shape();

  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  TensorProto proto;
  if (!can_memcpy) {
    in.AsProtoTensorContent(&proto);
  }
  size_t tensor_bytes = can_memcpy ? in.TotalBytes() : proto.ByteSize();
  if (is_dead) {
    tensor_bytes = 0;
  }
  entry->send_micros_ = 0;
  // prefix
  string prefix = DriverPrefixMessage::CreateDriverPrefixMessage(in.shape(),
      tensor_bytes, is_dead, entry->send_micros_, entry->meta_changed_);
  uint32_t imm_data = entry->uinque_id_;

  // tensor
  uint32_t send_tensor_lkey = 0;
  size_t prefix_s = prefix.size();
  int need_length = prefix_s + tensor_bytes;
  if (entry->tensor_addr_ == nullptr) {
    if (!FindLocalMr(name, &entry->tensor_addr_,
          &entry->smr_, &entry->local_allocate_size_)) {
      entry->local_allocate_size_ = 0;
    }
  }
  if (need_length > entry->local_allocate_size_) {
    LOG(INFO) << "key :" << name << " relloc need_length:"
              << need_length
              << " already size:"
              << entry->local_allocate_size_;
    entry->local_allocate_size_ = Alloc(prefix_s + 
        VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * tensor_bytes,
        &entry->tensor_addr_, &entry->smr_, false);
  }

  if (!is_dead) {
    if (can_memcpy) {
      // allocate region and copy data
      entry->src_buffer_ = const_cast<TensorBuffer*>(DMAHelper::buffer(&in));
      if (entry->src_buffer_ != nullptr) {
        if (tensor_bytes > 0) {
          void* addr_offset = (void*)((uint64_t)entry->tensor_addr_ + prefix_s);
          memcpy(addr_offset, DMAHelper::base(&in), tensor_bytes);
        }
      }
    } else {
      // for send dirven
      void* addr_offset = (void*)((uint64_t)entry->tensor_addr_ + prefix_s);
      proto.SerializeToArray(addr_offset, tensor_bytes);
    }
  } else {
    tensor_bytes = 0;
  }
  memcpy(entry->tensor_addr_, prefix.data(), prefix_s);
  send_tensor_lkey = (entry->smr_ == nullptr) ?
                        0 : entry->smr_->lkey;
  // remote mr addr
  uint64_t remote_addr = entry->addr_;
  uint32_t rkey = entry->lkey_;
  CHECK(tensor_bytes + prefix_s <= entry->allocate_size_)
      << " 1name:" << name
      << " May should large allocate static memory ratio"
      << " tensor_bytes:" << tensor_bytes
      << " prefix_s:" << prefix_s
      << " entry->allocate_size_:" << entry->allocate_size_;
  auto tensor_addr = (uint64_t)entry->tensor_addr_;
  RdmaMessageBuffer::Write(this, imm_data, tensor_bytes + prefix_s,
      tensor_addr, send_tensor_lkey, remote_addr, rkey,
      RDMA_WRITE_ID_SEND_DEIVER_WRITE, entry.get());
}

void RdmaChannel::InitAndSetDriverStatus() {
  size_t entries_size = rdma_send_driver_mgr_->InitLocalDriverEntry();
  // init LocalDriverBufferMgr
  size_t ready_size = local_driver_buffer_mgr_->InitLocalDriverBufferMgr();
  CHECK_EQ(entries_size, ready_size)
      << "NotifyAsyncAllocator entries_size:"
      << entries_size
      << " ready_size:"
      << ready_size;
  // TODO(wuyongyu) could_send_driver must set before Async InitLocalDriverEntry
  could_send_driver_ = true;
}

void RdmaChannel::PleSendOrCheck() {
  const auto& remote_name = remote_name_;
  const auto& local_name = local_name_;
  RDMA_LOG(1) << "NotifyRemoteDriverEntry local_worker_name:" << local_name
            << " remote_name:" << remote_name;

  // get the channel cache
  SharedGrpcChannelPtr client_channel =
      channel_cache_->FindWorkerChannel(remote_name);
  CHECK(client_channel != nullptr) << "PleSendOrCheck target:"
                                   << remote_name
                                   << " client_channel is null!";
  GrpcVerbsClient* client = new GrpcVerbsClient(client_channel);
  CHECK(client != nullptr) << "PleSendOrCheck No worker known as "
                           << remote_name;

  PleSendOrCheckReq req;
  req.set_host_name(local_name);
  // synchronous call
  PleSendOrCheckResp resp;
  Status s;
  int attempts = 0;
  static const int max_num_attempts = 5;
  do {
    s = client->ReqPleSendOrCheck(&req, &resp);
    // save obtained remote addresses
    // connect to the remote channel
    if (s.ok() && resp.is_ok()) {
      LOG(INFO) << "verbs to "<< remote_name << " ReqPleSendOrCheck succeed!";
    } else {
      LOG(ERROR) << "ReqPleSendOrCheck Connecting to "
                 << remote_name << ": Got "
                 << s.error_message() << ". Retrying (" << (attempts + 1)
                 << " Remote worker Async status:" << resp.is_ok()
                 << "/" << max_num_attempts << ")..."
                 << " resp.is_ok:" << resp.is_ok();

      if (++attempts == max_num_attempts) {
        CHECK(FATAL) << "RdmaChannel::PleSendOrCheck failed";
      }
      adapter_->worker_env_->env->SleepForMicroseconds(2000000);
    }
  } while (!s.ok());
  delete client;
}

void RdmaTensorResponse::SendBck(const Tensor& in, const TensorProto& proto,
                              bool is_dead, const Status& status) {
  if (!status.ok()) {
    SendErrorStatus(status, "RdmaTensorResponse::SendBck::!status.ok");
    return;
  }
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  bool proto_size_changed =
      (!can_memcpy) && (proto.ByteSize() != rm_.tensor_bytes_);

  int pair_index = (request_index_ % RdmaChannel::kNumMessageBuffers) / 2;
  int buffer_index = 2 * pair_index;
  auto* tx_buffer = channel_->message_buffers()[buffer_index];
  // move cpu allocator tensor to RdmaMR tensor
  // RdmaClone(in, proto, is_dead);
  if (meta_data_changed_ || proto_size_changed) {
    Clone(in, proto, is_dead);
    // Here is a bug
    SendMetaData(in, proto, is_dead);
    tx_buffer->SendNextItem();
  } else {
    SendContent(in, proto, is_dead, false);
  }
}

bool RdmaTensorResponse::TensorMetaDataChanged(const Tensor& in, bool is_dead) {
  return (rm_.data_type_ != in.dtype()) || (rm_.tensor_shape_ != in.shape()) ||
         (rm_.is_dead_ != is_dead);
}

void RdmaTensorResponse::RdmaClone(const Tensor& in, const TensorProto& proto,
                               bool is_dead) {
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  if (can_memcpy && (in.TotalBytes() > 0)) {
    tensor_ = new Tensor(channel_->rdma_mem_allocator_, in.dtype(), in.shape());
    memcpy(DMAHelper::base(tensor_), DMAHelper::base(&in), in.TotalBytes());
  } else {
    tensor_ = new Tensor(in.dtype(), in.shape());
  }
  if (!can_memcpy) {
    proto_ = new TensorProto(proto);
  }
  is_dead_ = is_dead;
}

void RdmaTensorResponse::Clone(const Tensor& in, const TensorProto& proto,
                               bool is_dead) {
  // Clone the data to be sent later. For simplicity, we clone the tensor's
  // data even if it is already a copy. Performance is less of a concern here
  // since the meta-data hardly ever changes. The reason we create a copy, is
  // that some tensors share their buffer between different step-ids, so the
  // tensor content may change before re-request was completed.
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  // if (can_memcpy && (in.TotalBytes() > 0)) {
  //   AllocatorAttributes host_alloc_attrs;
  //   host_alloc_attrs.set_nic_compatible(true);
  //   host_alloc_attrs.set_on_host(true);
  //   Allocator* allocator = src_dev_->GetAllocator(host_alloc_attrs);
  //   tensor_ = new Tensor(allocator, in.dtype(), in.shape());
  //   memcpy(DMAHelper::base(tensor_), DMAHelper::base(&in), in.TotalBytes());
  // } else {
  //   tensor_ = new Tensor(in.dtype(), in.shape());
  // }
  if (can_memcpy && (in.TotalBytes() > 0)) {
    channel_->FindOrCreateRemoteBytesAddrMemoryRegion(rm_.name_,
          &src_addr_, &mr_, &res_region_, in.TotalBytes());
    auto src_buffer = const_cast<TensorBuffer*>(DMAHelper::buffer(&in));
    memcpy(src_addr_, DMAHelper::base(&in), in.TotalBytes());
    res_fake_allocator_ = new FakeAllocator(src_addr_);
    tensor_ = new Tensor(res_fake_allocator_, in.dtype(), in.shape());
  } else {
    tensor_ = new Tensor(in.dtype(), in.shape());
  }
  if (!can_memcpy) {
    proto_ = new TensorProto(proto);
  }
  is_dead_ = is_dead;
}

void RdmaTensorResponse::SendMetaData(const Tensor& in,
                                      const TensorProto& proto, bool is_dead) {
  // LOG(INFO) << "SendMetaData...";
  send_meta_begin_ = 0;
  RDMA_LOG(2) << "Request #" << rm_.request_index_
              << ": Meta data changed: " << rm_.name_;
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  size_t tensor_bytes = (can_memcpy) ? in.TotalBytes() : proto.ByteSize();

  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_META_DATA_UPDATE;
  rm.name_size_ = rm_.name_.size();
  rm.name_ = rm_.name_;
  rm.tensor_shape_ = in.shape();
  rm.data_type_ = in.dtype();
  rm.step_id_ = rm_.step_id_;
  rm.is_dead_ = is_dead;
  rm.tensor_bytes_ = tensor_bytes;
  rm.request_index_ = rm_.request_index_;
#ifdef RDMA_DATA_VALIDATION
  rm.checksum_ = checksum_;
#endif
  RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
              << ": Sending RDMA_MESSAGE_META_DATA_UPDATE #"
              << rm.request_index_ << ": " << rm.name_
              << " (shape = " << rm.tensor_shape_.DebugString() << "."
              << " data-type = " << DataTypeString(rm.data_type_) << "."
              << " is-dead = " << rm.is_dead_ << ")";

  // rm.create_micros_ = send_meta_begin_;
  string message = RdmaMessage::CreateMessage(rm);
  int pair_index = (request_index_ % RdmaChannel::kNumMessageBuffers) / 2;
  int buffer_index = 2 * pair_index;
  auto* tx_message_buffer  = channel_->message_buffers()[buffer_index];
  tx_message_buffer->EnqueueItem(message);
}

void RdmaTensorResponse::SendContent(const Tensor& in, const TensorProto& proto,
                                     bool is_dead,
                                     bool is_resume) {
  // update recv_local_send_rmda
  // overcome the sendmeta effects
  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  size_t tensor_bytes = (can_memcpy) ? in.TotalBytes() : proto.ByteSize();
  uint32_t imm_data = rm_.request_index_;

  AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_nic_compatible(true);
    host_alloc_attrs.set_on_host(true);
  Allocator* allocator = src_dev_->GetAllocator(host_alloc_attrs);
  if (!is_dead) {
    if (can_memcpy && !is_resume || in.TotalBytes() == 0) {
      // when send_content directly so we need copy data
      src_buffer_ = const_cast<TensorBuffer*>(DMAHelper::buffer(&in));
      if (src_buffer_ != nullptr) {
        // src_buffer_->Ref();  // Keep buffer alive until write is complete
        // TODO(wuyongyu02): Move to Meta Change
        channel_->FindOrCreateRemoteBytesAddrMemoryRegion(rm_.name_,
            &src_addr_, &mr_, &res_region_, tensor_bytes);
        if (tensor_bytes > 0) {
          memcpy(src_addr_, src_buffer_->data(), tensor_bytes);
        }
      }
    } 
    if(!can_memcpy){
      RDMA_LOG(2) << "Encoding proto: " << rm_.name_
                  << " (Size: " << tensor_bytes << ") " << in.DebugString();
      channel_->FindOrCreateRemoteBytesAddrMemoryRegion(rm_.name_,
          &src_addr_, &mr_, &res_region_, tensor_bytes);
      proto.SerializeToArray(src_addr_, tensor_bytes);
    }
  } else {
    tensor_bytes = 0;
  }

  uint32_t lkey = (mr_ == nullptr) ? 0 : mr_->lkey;
  RDMA_LOG(1) << "Step 0x" << std::hex << rm_.step_id_ << std::dec
              << ": Sending tensor content #" << rm_.request_index_ << " from "
              << std::hex << src_addr_ << " (0x" << lkey << ")"
              << " to " << rm_.remote_addr_ << " (0x" << rm_.rkey_
              << "): " << rm_.name_ << " (size: 0x" << std::hex << tensor_bytes
              << ")";

  RdmaMessageBuffer::Write(channel_, imm_data, tensor_bytes,
                           (uint64_t)src_addr_, lkey, rm_.remote_addr_,
                           rm_.rkey_, RDMA_WRITE_ID_TENSOR_WRITE, this);
}

void RdmaTensorResponse::SendErrorStatus(const Status& status,
    const std::string& src_func_name) {
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_ERROR_STATUS;
  rm.name_size_ = rm_.name_.size();
  rm.name_ = rm_.name_;
  rm.step_id_ = rm_.step_id_;
  rm.request_index_ = rm_.request_index_;
  rm.status_ = status;

  LOG(INFO) << "Step 0x" << (int64)rm.step_id_ << std::dec
             << ": Sending RDMA_MESSAGE_ERROR_STATUS #" << rm.request_index_
             << ": " << rm.name_ << ". Status: " << status.ToString()
             << " src_func_name:" << src_func_name;

  string message = RdmaMessage::CreateMessage(rm);

  int pair_index = (request_index_ % RdmaChannel::kNumMessageBuffers) / 2;
  int buffer_index = 2 * pair_index;
  // buffer_index = 0;
  auto* tx_message_buffer  = channel_->message_buffers()[buffer_index];
  // channel_->tx_message_buffer_->EnqueueItem(message);
  // channel_->tx_message_buffer_->SendNextItem();
  tx_message_buffer->EnqueueItem(message);
  tx_message_buffer->SendNextItem();
  // Destroy the response.
  Destroy();
}

void RdmaTensorResponse::Destroy() {
  if (res_region_.get() != nullptr) {
    // res_region_->Unref();
  }
  // response end
  if (src_buffer_ != nullptr) {
    // src_buffer_->Unref();
  }
  if (tensor_ != nullptr) {
    delete tensor_;
  }
  if (proto_ != nullptr) {
    // ibv_dereg_mr(mr_);
    // free(src_addr_);
    delete proto_;
  }
  // Remove response from the pending list:
  channel_->RemoveTensorResponse(rm_.request_index_);
}

// Create a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure
// Returns:
//   message in string format
string RdmaMessage::CreateMessage(const RdmaMessage& rm) {
  // Rdma Message format
  // type|name_size|name|step_id|request_index|remote_addr|rkey|is_dead|...
  //   1B|    2B   | 512|  8B   |     8B      |       8B  | 4B |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|create_micros|error_status       |
  // ...|   XB    |    XB      |    8B      |8B        |size - 4B, proto - XB |
  // ...| remote_bytes_addr    | remote_bytes_value
  //         8B                |    4B
  //
  // ACK:             Imm-type: ACK
  // TENSOR_REQUEST:  Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, remote_addr,
  //                      rkey, is_dead, data_type, tensor_shape, tensor_bytes
  // META_DATA_UPDATE: Imm-type: MESSAGE
  //                  Fields: type, request_index, is_dead, data_type,
  //                      tensor_shape, tensor_bytes
  // TENSOR_RE_REQUST: Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, remote_addr,
  //                      rkey, is_dead, data_type, tensor_shape, tensor_bytes
  // ERROR_STATUS:    Imm-type: MESSAGE
  //                  Fields: type, request_index, name, step_id, error_status
  // Tensor content:  Imm-type: request_index
  size_t message_size = kMessageTotalBytes;
  char message[kMessageTotalBytes + kErrorStatusMaxSize + 100];
  // type
  message[kTypeStartIndex] = static_cast<char>(rm.type_) & 0xff;
  // request index
  memcpy(&message[kRequestIndexStartIndex], &rm.request_index_,
         sizeof(rm.request_index_));

  // name, step_id, remote_addr, rkey
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&message[kNameSizeStartIndex], &rm.name_size_,
           sizeof(rm.name_size_));
    memcpy(&message[kNameStartIndex], rm.name_.data(), rm.name_.size());
    memcpy(&message[kRemoteAddrStartIndex], &rm.remote_addr_,
           sizeof(rm.remote_addr_));
    memcpy(&message[kRkeyStartIndex], &rm.rkey_, sizeof(rm.rkey_));
    memcpy(&message[kStepIdStartIndex], &rm.step_id_, sizeof(rm.step_id_));

    // memcpy(&message[KRemoteBytesAddrKeyStartIndex],
    //          &rm.remote_bytes_addr_key_, sizeof(rm.remote_bytes_addr_key_));
    // memcpy(&message[KRemoteBytesAddrStartIndex],
    //          &rm.remote_bytes_addr_, sizeof(rm.remote_bytes_addr_));
  }
  // is_dead, data_type, tensor_shape, tensor_bytes, create_micros
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) ||
      (rm.type_ == RDMA_MESSAGE_DRIVER_BEGIN) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&message[kIsDeadStartIndex], &rm.is_dead_, sizeof(rm.is_dead_));

    memcpy(&message[kDataTypeStartIndex], &rm.data_type_,
           sizeof(rm.data_type_));
    memcpy(&message[kTensorShapeStartIndex], &rm.tensor_shape_,
           sizeof(rm.tensor_shape_));
    memcpy(&message[kTensorBytesStartIndex], &rm.tensor_bytes_,
           sizeof(rm.tensor_bytes_));
    // memcpy(&message[kCreateMicrosStartIndex], &rm.create_micros_,
    //       sizeof(rm.create_micros_));
  }
  // checksum
#ifdef RDMA_DATA_VALIDATION
  memcpy(&message[kChecksumStartIndex], &rm.checksum_, sizeof(rm.checksum_));
#endif
  // error status
  if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
    ::grpc::Status gs = ToGrpcStatus(rm.status_);
    // (wuyongyu) decrease the error message size https://km.sankuai.com/page/403000580
    // ::grpc::Status gs = ::grpc::Status::OK;
    ErrorStatusProto gsProto;
    gsProto.set_error_code(gs.error_code());
    gsProto.set_error_message(gs.error_message());
    gsProto.set_error_details(gs.error_details());
    uint32_t gsProtoSize = gsProto.ByteSize();
    if (gsProtoSize + 4 > kErrorStatusMaxSize) {
      LOG(ERROR) << "Error status (" << gsProtoSize + 4 << " bytes) "
                 << "is too big to fit in RDMA message (" << kErrorStatusMaxSize
                 << " bytes). Truncated.";
      gsProtoSize = kErrorStatusMaxSize - 4;
    }
    uint32_t* proto_size = (uint32_t*)&message[kErrorStatusStartIndex];
    *proto_size = gsProtoSize;
    gsProto.SerializeToArray(&message[kErrorStatusStartIndex + 4], gsProtoSize);
    message_size += gsProtoSize + 4;
  }
  return string(message, message_size);
}

string FussionMessages::CreateFusionMessages(
    const std::vector<RdmaMessage>& rmv) {
  CHECK(rmv.size() < kRdmaMaxMessagesNumber)
      << "FussionMessages CreateFusionMessages must less "
      << kRdmaMaxMessagesNumber;
  size_t message_size = kTotalFussionMessageSize;
  char message[kTotalFussionMessageSize + RdmaMessage::kRdmaMessageBufferSize];
  uint32_t* mn = (uint32_t*)&message[kMessageNumbersStartIndex];
  *mn = rmv.size();
  for (int i = 0; i < rmv.size(); i++) {
    string m = RdmaMessage::CreateMessage(rmv[i]);
    uint32_t* ms = (uint32_t*)&message[kMessageSizeStartIndex + i * 4];
    *ms = m.size();
    uint32_t s;
    memcpy(&s, &message[kMessageSizeStartIndex + i * 4], sizeof(s));
    memcpy(&message[KStringMessagesStartIndex +
           i * RdmaMessage::kRdmaMessageBufferSize], m.data(), m.size());
  }
}

void FussionMessages::ParseFussionMessages(std::vector<RdmaMessage>& rmv,
    void* buffer) {
  char* message = static_cast<char*>(buffer);
  uint32_t mn = 0;
  memcpy(&mn, &message[kMessageNumbersStartIndex], sizeof(mn));
  if (mn == 0) {
    return;
  }
  rmv.reserve(mn);
  for (int i=0; i < mn; i++) {
    uint32_t message_size;
    memcpy(&message_size, &message[kMessageSizeStartIndex + i * 4],
           sizeof(message_size));
    char m[RdmaMessage::kMessageTotalBytes +
            RdmaMessage::kErrorStatusMaxSize + 100];
    memcpy(m, &message[KStringMessagesStartIndex +
            i * RdmaMessage::kRdmaMessageBufferSize], message_size);
    RdmaMessage::ParseMessage(rmv[i], &m);
  }
}

void RdmaMessage::ParseMessage(RdmaMessage& rm, void* buffer) {
  char* message = static_cast<char*>(buffer);
  // type
  rm.type_ = static_cast<RdmaMessageType>(message[kTypeStartIndex]);
  // request index
  memcpy(&rm.request_index_, &message[kRequestIndexStartIndex],
         sizeof(rm.request_index_));

  // name, step_id, remote_addr, rkey
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&rm.name_size_, &message[kNameSizeStartIndex],
           sizeof(rm.name_size_));
    rm.name_ = string(&message[kNameStartIndex], rm.name_size_);
    memcpy(&rm.remote_addr_, &message[kRemoteAddrStartIndex],
           sizeof(rm.remote_addr_));
    memcpy(&rm.rkey_, &message[kRkeyStartIndex], sizeof(rm.rkey_));
    memcpy(&rm.step_id_, &message[kStepIdStartIndex], sizeof(rm.step_id_));
    // memcpy(&rm.remote_bytes_addr_key_,
    //          &message[KRemoteBytesAddrKeyStartIndex],
    //          sizeof(rm.remote_bytes_addr_key_));
    // memcpy(&rm.remote_bytes_addr_,
    //          &message[KRemoteBytesAddrStartIndex],
    //          sizeof(rm.remote_bytes_addr_));
  }
  // data_type, tensor_bytes, tensor_shape, is_dead
  if ((rm.type_ == RDMA_MESSAGE_TENSOR_REQUEST) ||
      (rm.type_ == RDMA_MESSAGE_META_DATA_UPDATE) ||
      (rm.type_ == RDMA_MESSAGE_DRIVER_BEGIN) ||
      (rm.type_ == RDMA_MESSAGE_TENSOR_RE_REQUEST)) {
    memcpy(&rm.is_dead_, &message[kIsDeadStartIndex], sizeof(rm.is_dead_));
    memcpy(&rm.data_type_, &message[kDataTypeStartIndex],
           sizeof(rm.data_type_));
    memcpy(&rm.tensor_shape_, &message[kTensorShapeStartIndex],
           sizeof(rm.tensor_shape_));
    memcpy(&rm.tensor_bytes_, &message[kTensorBytesStartIndex],
           sizeof(rm.tensor_bytes_));
    // memcpy(&rm.create_micros_, &message[kCreateMicrosStartIndex],
    //       sizeof(rm.create_micros_));
  }
  // checksum
#ifdef RDMA_DATA_VALIDATION
  memcpy(&rm.checksum_, &message[kChecksumStartIndex], sizeof(rm.checksum_));
#endif
  // error status
  if (rm.type_ == RDMA_MESSAGE_ERROR_STATUS) {
    ErrorStatusProto gsProto;
    uint32_t gsProtoSize = *(uint32_t*)&message[kErrorStatusStartIndex];
    CHECK(ParseProtoUnlimited(&gsProto, &message[kErrorStatusStartIndex + 4],
                              gsProtoSize))
        << "Failed to parse error status proto from message. Aborting.";
    ::grpc::Status gs((::grpc::StatusCode)gsProto.error_code(),
                      gsProto.error_message(), gsProto.error_details());
    rm.status_ = FromGrpcStatus(gs);
  }
}

ibv_mr* RdmaChannel::FindMemoryRegion(void* addr, size_t length) {
  return rdma_memory_mgr_->FindMemoryRegion(addr, length);
}

//*****************************************************************************
// RdmaMemoryMgr
//*****************************************************************************

ibv_mr* RdmaMemoryMgr::FindMemoryRegion(void* addr, size_t length) {
  mutex_lock l(mrs_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter == std::end(mrs_) || iter->get()->addr > addr) {
    return nullptr;
  } else {
    return iter->get();
  }
}

void RdmaMemoryMgr::InsertMemoryRegion(void* addr, size_t length,
                                       const std::string& allocator_name) {
  if (length == 0) return;
  ibv_mr* mr = ibv_reg_mr(pd_, addr, length,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  RDMA_LOG(1) << "Insert memory region 0x" << std::hex << mr->rkey << ". ["
              << addr << "-" << (void*)((uint64_t)addr + length - 1) << "]"
              << " SIZE: 0x" << length << " (" << allocator_name << ").";
  // LOG(INFO) << "Insert memory region 0x" << std::hex << mr->rkey << ". ["
  //           << addr << "-" << (void*)((uint64_t)addr + length - 1) << "]"
  //           << " SIZE: 0x" << length << " (" << allocator_name << ").";
  if (mr != nullptr) {
    mutex_lock l(mrs_mu_);
    auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
    mrs_.insert(iter, {mr, &MRDeleter});
  } else {
    LOG(FATAL) << "Cannot register memory region";
  }
}

void RdmaMemoryMgr::EvictMemoryRegion(void* addr, size_t length) {
  if (length == 0) return;
  mutex_lock l(mrs_mu_);
  auto iter = std::upper_bound(mrs_.begin(), mrs_.end(), addr, &Comparator);
  if (iter != std::end(mrs_) && iter->get()->addr == addr) {
    mrs_.erase(iter);
    RDMA_LOG(1) << "Evict memory region 0x" << std::hex << iter->get()->rkey;

  } else {
    LOG(WARNING) << "Failed to de-register memory region";
  }
}

const TensorMetaData* RdmaChannel::GetTensorMetaData(
    const std::string& tensor_name) {
  mutex_lock l(tensor_meta_data_mu_);
  auto it = tensors_meta_data_.find(tensor_name);
  if (it == tensors_meta_data_.end()) {
    return nullptr;
  }
  return &it->second;
}

const TensorMetaData* RdmaChannel::SetTensorMetaData(
    const std::string& tensor_name, DataType dtype, const TensorShape& shape,
    bool is_dead, size_t proto_size) {
  mutex_lock l(tensor_meta_data_mu_);
  TensorMetaData& meta_data = tensors_meta_data_[tensor_name];
  meta_data.data_type_ = dtype;
  meta_data.tensor_shape_ = shape;
  meta_data.proto_size_ = proto_size;
  meta_data.is_dead_ = is_dead;
  return &meta_data;
}

//*****************************************************************************
// RdmaTensorRequest
//*****************************************************************************

Status LocalDriverBufferMgr::QueueRdmaSave(const string& key,
    const Args& send_args, Tensor* val, const bool is_dead,
    const uint64& send_begin_micros) {
  string key_hash(key);
  if (!status_.ok()) {
    Status s = status_;
    return s;
  }
  QueueItems* queue_pair = queue_table_[key_hash];
  CHECK(queue_pair != nullptr) << "QueueRdmaSave queue_pair is nullptr:"
                               << key_hash;
  ItemQueue * queue_item = queue_pair->queue;
  queue_pair->queue_lock_.lock();
  if (queue_item->empty() || queue_item->front()->HasValue()) {
    RDMA_LOG(1) << "QueueRdmaSave Enqueue Send Item (key:" << key << "). ";
    Item* item = new Item;
    item->value = val;
    item->is_dead = is_dead;
    item->has_value = true;
    item->send_args = send_args;
    item->send_start_micros_ =  Env::Default()->NowMicros();
    if (item->send_args.device_context) {
      item->send_args.device_context->Ref();
    }
    queue_item->push_back(item);
    // LOG(INFO) << "QueueRdmaEnqueueSendWaitRecv_Micros:"
    //              << item->send_start_micros_ - send_args.rendezvous_micros;
    queue_pair->queue_lock_.unlock();
    return Status::OK();
  }
  RDMA_LOG(1) << "QueueRdmaSave Consume Recv Item (key:" << key << "). ";
  Item* item = queue_item->front();
  if (queue_item->size() == 1) {
    VLOG(2) << "Clean up Send/Recv queue (key:" << key << "). ";
    // queue_table_.erase(key_hash);
    queue_item->pop_front();
  } else {
    queue_item->pop_front();
  }
  queue_pair->queue_lock_.unlock();
  DCHECK(item->HasCallback());
  // LOG(INFO) << "QueueRdmaRecvWaitSend_Micros key:" << key << " micros:"
  //           << Env::Default()->NowMicros() - item->recv_start_micros_;
  item->waiter(Status::OK(), send_args, item->recv_args, *val, is_dead);
  delete item;
  return Status::OK();
}

Status LocalDriverBufferMgr::RdmaSave(const string& key, const Args& send_args,
    const Tensor& val, const bool is_dead) {
  LOG(FATAL) << "this should not used;";
  return Status::OK();
}

void LocalDriverBufferMgr::QueueLoadAsync(const string& key,
    const Args& recv_args, DoneCallback done,
    const uint64& request_start_micros) {
  string key_hash(key);
  if (!status_.ok()) {
    // Rendezvous has been aborted.
    Status s = status_;
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }
  const auto& find = queue_table_.find(key_hash);
  if (find == queue_table_.end()) {
    for (auto& find : queue_table_) {
      if (absl::StrContains(key_hash, find.first)) {
        key_hash = find.first;
        break;
      }
    }
  }
  QueueItems* queue_pair = queue_table_[key_hash];
  CHECK(queue_pair != nullptr)
      << "QueueLoadAsync queue_pair is null:" << key_hash;
  ItemQueue * queue_item = queue_pair->queue;

  queue_pair->queue_lock_.lock();
  if (queue_item->empty() || !queue_item->front()->HasValue()) {
    CancellationManager* cm = recv_args.cancellation_manager;
    CancellationToken token = CancellationManager::kInvalidToken;
    bool already_cancelled = false;
    if (cm != nullptr) {
        token = cm->get_cancellation_token();
        already_cancelled = !cm->RegisterCallback(token, [this, token,
                                                          key_hash] {
          Item* item = nullptr;
          {
            QueueItems* queue_pair = queue_table_[key_hash];
            ItemQueue * queue_item = queue_pair->queue;
            if (queue_item->empty() || !queue_item->front()->HasValue()) {
              for (auto it = queue_item->begin(); it != queue_item->end();
                   it++) {
                if ((*it)->cancellation_token == token) {
                  item = *it;
                  if (queue_item->size() == 1) {
                    // key_hash queue can reuse
                    // table_.erase(key_hash);
                    queue_item->erase(it);
                  } else {
                    queue_item->erase(it);
                  }
                }
              }
            }
          }
          if (item != nullptr) {
            item->waiter(StatusGroup::MakeDerived(
                             errors::Cancelled("LoadAsync is cancelled.")),
                         Args(), item->recv_args, Tensor(), /*is_dead=*/false);
            delete item;
          }
      });
    }
    if (already_cancelled) {
      queue_pair->queue_lock_.unlock();
      done(StatusGroup::MakeDerived(
                 errors::Cancelled("LoadAsync is cancelled.")),
             Args(), recv_args, Tensor(), /*is_dead=*/false);
      return;
    }
    RDMA_LOG(1) << "LoadAsync Enqueue Recv Item (key:" << key << "). ";
    Item* item = new Item;
    if (cm != nullptr) {
      auto wrapped_done = std::bind(
          [cm, token](const DoneCallback& done,
                      // Begin unbound arguments.
                      const Status& s, const Args& send_args,
                      const Args& recv_args, const Tensor& v, bool dead) {
            cm->TryDeregisterCallback(token);
            RDMA_LOG(1) << "LoadAsync Enqueue Recv DoneCallback begin...";
            done(s, send_args, recv_args, v, dead);
          },
          std::move(done), std::placeholders::_1, std::placeholders::_2,
          std::placeholders::_3, std::placeholders::_4,
          std::placeholders::_5);
      item->waiter = std::move(wrapped_done);
    } else {
      item->waiter = std::move(done);
    }
    item->recv_args = recv_args;
    item->cancellation_token = token;
    item->request_start_micros_ = request_start_micros;
    item->recv_start_micros_ = Env::Default()->NowMicros();
    if (item->recv_args.device_context) {
      item->recv_args.device_context->Ref();
    }
    queue_item->push_back(item);
    queue_pair->queue_lock_.unlock();
    return;
  }
  RDMA_LOG(1) << "LoadAsync Consume Send Item (key:" << key << "). ";
  Item* item = queue_item->front();
  // LOG(INFO) << "QueueRdmaSendWaitRecv_Micros key:" << key << " micros:"
  //           << Env::Default()->NowMicros() - item->send_start_micros_;
  if (queue_item->size() == 1) {
    VLOG(2) << "Clean up Send/Recv queue (key:" << key << "). ";
    // queue_table_.erase(key_hash);
    queue_item->pop_front();
  } else {
    queue_item->pop_front();
  }
  queue_pair->queue_lock_.unlock();
  DCHECK(item->HasValue());
  done(Status::OK(), item->send_args, recv_args, *(item->value), item->is_dead);
  delete item;
}

void LocalDriverBufferMgr::LoadAsync(const string& key, const Args& recv_args,
                DoneCallback done) {
  LOG(FATAL) << "LoadAsync is not impl.";
  return;
}

size_t LocalDriverBufferMgr::InitLocalDriverBufferMgr() {
  RDMA_LOG(1) << "InitLocalDriverBufferMgr begin...";
  const auto& tensors_meta_data =
      channel_->channel_record_->GetChannelTensorsMetaData();
  const auto& tensors_uid_parsed_key =
      channel_->channel_record_->GetChannelTensorsUidParsedkey();

  CHECK(tensors_meta_data.size() == tensors_uid_parsed_key.size())
        << "tensors_meta_data size:"
        << tensors_meta_data.size()
        << " tensors_uid_parsed_key size:"
        << tensors_uid_parsed_key.size();

  std::vector<string> print_keys;
  for (auto& it : tensors_meta_data) {
    auto tfi = table_.find(it.first);
    if (tfi == table_.end()) {
      table_[it.first] = new Item();
    }
    auto qfi = queue_table_.find(it.first);
    if (qfi == queue_table_.end()) {
      print_keys.emplace_back(it.first);
      queue_table_[it.first] = new QueueItems();
      queue_table_[it.first]->queue = new ItemQueue();
    }
  }
  RDMA_LOG(1) << "InitLocalDriverBufferMgr Queutable:"
            << print_keys.size()
            << " "
            << absl::StrJoin(print_keys, ",");
  size_t ready_size = queue_table_.size();
  RDMA_LOG(1) << "InitLocalDriverBufferMgr end size:" << ready_size;
  return ready_size;
}

void LocalDriverBufferMgr::StartAbort(const Status& status) {
  CHECK(!status.ok());
  Table table;
  {
    status_.Update(status);
    table_.swap(table);
  }
  for (auto& p : table) {
    Item* item = p.second;
    if (!item->HasCallback()) {
      item->waiter(status, Args(), Args(), Tensor(), false);
    }
  }
}

//*****************************************************************************
// RdmaTensorRequest
//*****************************************************************************

RdmaTensorRequest::RdmaTensorRequest(
    uint32_t index, const string& key, int64 step_id, RdmaChannel* channel,
    Device* dst_dev, const Rendezvous::Args recv_args,
    const RdmaTensorRequest::RecvDoneCallback& done)
    : index_(index),
      step_id_(step_id),
      channel_(channel),
      dst_dev_(dst_dev),
      recv_args_(recv_args),
      result_tensor_(nullptr),
      proxy_tensor_(nullptr),
      rdma_addr_(nullptr),
      mr_(nullptr),
      done_(done),
      begin_start_req_(0) {
        key_.assign(key, 0, RdmaMessage::kNameCapacity);
}

RdmaTensorRequest::~RdmaTensorRequest() {
  DeallocateTensors();
}

void RdmaTensorRequest::Done(const Status& s) {
  Tensor val = std::move(*result_tensor_);
  Rendezvous::Args recv_args = std::move(recv_args_);
  bool is_dead = (meta_data_ == nullptr) ? false : meta_data_->is_dead_;
  RecvDoneCallback done = done_;
  DeallocateTensors();
  // if (result_region_.get() != nullptr) {
  //   result_region_->Unref();
  // }
  channel_->RemoveTensorRequest(index_);
  done(s, Rendezvous::Args(), recv_args, val, is_dead);
}

void RdmaTensorRequest::DeallocateTensors() {
  // if (fake_allocator_ != nullptr) {
  //   LOG(INFO) << "delete fake_allocator";
  //   delete fake_allocator_;
  //   fake_allocator_ = nullptr;
  // }
  if (result_tensor_ != nullptr) {
    delete result_tensor_;
    result_tensor_ = nullptr;
  }
  if (proxy_tensor_ != nullptr) {
    delete proxy_tensor_;
    proxy_tensor_ = nullptr;
  }
}

size_t RdmaChannel::Alloc(size_t size, void** p, ibv_mr** mr, 
                          bool dynamic, size_t realloc_size) const {
  size_t allocate_size = size;
  if (dynamic) {
    ib_malloc(p, &allocate_size, size, EIGEN_MAX_ALIGN_BYTES);
    *mr = ibv_reg_mr(pd_, *p, allocate_size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    return allocate_size;
  }
  // chunk alloc
  adapter_->recv_chunk_->Alloc(ib_allocate_size(size), p, mr, realloc_size);
  return allocate_size;
}

bool RdmaChannel::FindLocalMr(const std::string& key,
    void** remote_bytes_addr, ibv_mr** mr, int* length) {
  mutex_lock l(remote_bytes_addr_mu_);
  auto it = remote_bytes_addr_mrs_.find(key);
  if (it == remote_bytes_addr_mrs_.end()) {
   return false;
  }
  *remote_bytes_addr = it->second->addr_;
  *mr = it->second->mr_ptr_;
  *length = it->second->size_;
  CHECK(*remote_bytes_addr != nullptr && *mr != nullptr)
      << "key " << key << "*remote_bytes_addr is null?";
  return *remote_bytes_addr != nullptr && *mr != nullptr;
}

void RdmaChannel::FindOrCreateRemoteBytesAddrMemoryRegion(
    const std::string& key,
    void** remote_bytes_addr,
    ibv_mr** mr,
    std::shared_ptr<RemoteBytesAddrMemoryRegion> * region,
    size_t length,
    const Allocator* alloc_attr) {
  int allocate_size = 0;
  // region has already exists addr's info.
  if ((*region).get() != nullptr && (*region)->size_ > length) {
    *remote_bytes_addr = (*region)->addr_;
    *mr = (*region)->mr_ptr_;
    // (*region)->Ref();
    return;
  }
  // allocate_size = VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * length;
  // allocate_size = Alloc(allocate_size, remote_bytes_addr, mr, true);
  // *region = std::make_shared<RemoteBytesAddrMemoryRegion>(
  //                                     *remote_bytes_addr, *mr, allocate_size);
  // return;

  // TODO(wuyongyu02): KV is used,
  // because https://km.sankuai.com/page/641262306
  // because sparse tensors, so we need malloc large memory
  if (!could_send_driver_) {
    remote_bytes_addr_mu_.lock();
  }
  auto it = remote_bytes_addr_mrs_.find(key);
  if (it == remote_bytes_addr_mrs_.end()) {
    allocate_size = VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * length;
    // because concat DriverPrefixMessage
    allocate_size += DriverPrefixMessage::kPrefixMessageTotalBytes;
    allocate_size = Alloc(allocate_size, remote_bytes_addr, mr, false);
    *region = std::make_shared<RemoteBytesAddrMemoryRegion>(
        *remote_bytes_addr, *mr, allocate_size);
    remote_bytes_addr_mrs_[key] = *region;
    // LOG(INFO) << "#1 key:" << key << " size:" << length;
    if (!could_send_driver_) {
      remote_bytes_addr_mu_.unlock();
    }
  } else {
    if (length > it->second->size_) {
      allocate_size = VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * length;
      // because concat DriverPrefixMessage
      allocate_size += DriverPrefixMessage::kPrefixMessageTotalBytes;
      allocate_size = Alloc(allocate_size, remote_bytes_addr, mr, false, it->second->size_);
      *region = std::make_shared<RemoteBytesAddrMemoryRegion>(
          *remote_bytes_addr, *mr, allocate_size);
      if (length > it->second->size_) {
        it->second = *region;
        it->second->size_ = allocate_size;
      }
      // LOG(INFO) << "#2 create new tensor:" << key;
    } 
    // else if(it->second->RefCountIsOne()) {
    //   allocate_size = VerbsEnvRegistrar::Instance()->RdmaTensorBufferRatio() * length;
    //   // because concat DriverPrefixMessage
    //   allocate_size += DriverPrefixMessage::kPrefixMessageTotalBytes;
    //   allocate_size = Alloc(allocate_size, remote_bytes_addr, mr, false);
    //   *region = std::make_shared<RemoteBytesAddrMemoryRegion>(
    //       *remote_bytes_addr, *mr, allocate_size);
    // } 
    else {
      *region = it->second;
      *remote_bytes_addr = it->second->addr_;
      *mr = it->second->mr_ptr_;
      // LOG(INFO) << "#3 key:" << key << " size:" << length;
    }
    // (*region)->Ref();
    if (!could_send_driver_) {
      remote_bytes_addr_mu_.unlock();
    }
  }
}

size_t RdmaChannel::ChannelAllocateTensors(
    const string& key,
    const TensorMetaData& meta,
    const Allocator* alloc_attr, ibv_mr** mr/*new */,
    std::shared_ptr<RemoteBytesAddrMemoryRegion> * region,
    void** rdma_addr/*new*/) {
  size_t max_length = 0;
  if (DataTypeCanUseMemcpy(meta.data_type_)) {
    max_length = RecordTensorMetaData::GetTensorLength(meta.data_type_,
                                                        meta.tensor_shape_);
  } else {
    max_length = meta.proto_size_;
  }
  // use allocator for RdmaTensorrequest
  FindOrCreateRemoteBytesAddrMemoryRegion(key, rdma_addr, mr, region,
                                          max_length, alloc_attr);
  return max_length;
}

size_t RdmaTensorRequest::GetTensorLength(const TensorMetaData& meta) {
  size_t max_length = 0;
  if (DataTypeCanUseMemcpy(meta.data_type_)) {
    max_length = RecordTensorMetaData::GetTensorLength(meta.data_type_,
                                                        meta.tensor_shape_);
  } else {
    max_length = meta.proto_size_;
  }
  return max_length;
}

bool RdmaTensorRequest::AllocateTensors() {
  auto len = channel_->ChannelAllocateTensors(
      key_, *meta_data_, dst_dev_->GetAllocator(recv_args_.alloc_attrs),
      &mr_, &result_region_, &rdma_addr_);
  if (DataTypeCanUseMemcpy(meta_data_->data_type_)) {
    fake_allocator_ = new FakeAllocator(rdma_addr_);
    result_tensor_ = new Tensor(fake_allocator_,
                               meta_data_->data_type_, 
                               meta_data_->tensor_shape_);
  } else {
    // proto
    result_tensor_ =
      new Tensor(dst_dev_->GetAllocator(recv_args_.alloc_attrs),
                 meta_data_->data_type_, meta_data_->tensor_shape_);
  }
  size_t tensor_size = result_tensor_->TotalBytes();
  bool can_memcpy = DataTypeCanUseMemcpy(result_tensor_->dtype());
  if (can_memcpy && tensor_size == 0) {
    return true;
  }
#if GOOGLE_CUDA
    if (can_memcpy) {
      // Can't RDMA directly to result. Use a proxy.
      proxy_tensor_ =
          new Tensor(GPUProcessState::singleton()->GetGpuHostAllocator(0),
                     result_tensor_->dtype(), result_tensor_->shape());
      rdma_addr_ = DMAHelper::base(proxy_tensor_);
      // mr_ =
      //     RdmaMemoryMgr::Singleton().FindMemoryRegion(rdma_addr_, tensor_size);
    }
#endif
  CHECK(mr_ != nullptr) << " No memory region found for address " 
                        << rdma_addr_
                        << ": " << key_;
  return true;
}

void RdmaTensorRequest::AllocateTensorsAsync(StatusCallback done) {
  AllocateTensors();
  bool on_host = recv_args_.alloc_attrs.on_host();
  if (dst_dev_->tensorflow_gpu_device_info() && !on_host &&
      (proxy_tensor_ == nullptr)) {
#if GOOGLE_CUDA
    // We need to sync the memory allocation on the GPU:
    StreamGPUOp(dst_dev_, recv_args_.device_context, done);
#endif
  } else {
    done(Status::OK());
  }
}

void RdmaTensorRequest::Send(RdmaMessageType message_type) {
  int pair_index = (index_ % RdmaChannel::kNumMessageBuffers) / 2;
  int buffer_index = 2 * pair_index;
  auto* rb  = channel_->message_buffers()[buffer_index];
  RdmaMessage rm;
  rm.type_ = message_type;
  rm.request_index_ = index_;
  rm.name_size_ = key_.size();
  rm.name_ = key_;
  rm.step_id_ = step_id_;
  rm.remote_addr_ = (uint64_t)rdma_addr_;
  if (meta_data_ != nullptr) {
    rm.data_type_ = meta_data_->data_type_;
    rm.tensor_shape_ = meta_data_->tensor_shape_;
    rm.is_dead_ = meta_data_->is_dead_;
    rm.tensor_bytes_ = meta_data_->proto_size_;
  } else {
    rm.data_type_ = DT_INVALID;
  }
  rm.rkey_ = (mr_ == nullptr) ? 0 : mr_->rkey;
  // rm.create_micros_ = 0;
  RDMA_LOG(1) << "Step 0x" << std::hex << rm.step_id_ << std::dec
              << ": Sending  " << MessageTypeToString(message_type) << " #"
              << index_ << ": " << rm.name_ << " on " << rdma_addr_
              << " (rkey: 0x" << std::hex << rm.rkey_ << ")";

  string message = RdmaMessage::CreateMessage(rm);
  rb->EnqueueItem(message);
  rb->SendNextItem();
}

void RdmaTensorRequest::RecvTensorMetaData(DataType dtype, TensorShape shape,
                                           bool is_dead, size_t proto_size) {
  meta_data_ = channel_->SetTensorMetaData(
      key_, dtype, shape, is_dead, proto_size);
  // channel record MetaData
  channel_->channel_record_->Record(key_, *meta_data_);
  // global record
  // RecordTensorMetaData::Singleton().GlobalRecord(key_, *meta_data_);
  DeallocateTensors();
  // if (result_region_.get() != nullptr) {
  //   result_region_->Unref();
  // }
  AllocateTensorsAsync(
      [this](const Status& s) { Send(RDMA_MESSAGE_TENSOR_RE_REQUEST); });
}

void RdmaTensorRequest::RecvTensorContent() {
  uint64_t deal_data_begin = Env::Default()->NowMicros();
  bool can_memcpy = DataTypeCanUseMemcpy(meta_data_->data_type_);
  size_t message_size =
      can_memcpy ? result_tensor_->TotalBytes() : meta_data_->proto_size_;

  RDMA_LOG(1) << "Step 0x" << std::hex << step_id_ << std::dec
              << ": Received tensor content #" << index_ << ": " << key_
              << " (Size: 0x" << std::hex << message_size << ")";

  if (can_memcpy) {
    // copy Tensor from rdma_addr_
    // TODO(wuyongyu)
    // only the rdma_addr_ has value , can memcpy
    // if (result_tensor_->TotalBytes() > 0) {
    //   memcpy(DMAHelper::base(result_tensor_), (void*)(rdma_addr_),
    //         result_tensor_->TotalBytes());
    // }
    // Recv Tensor memory if can resuse
    Done(Status::OK());
  } else {
    TensorProto proto;
    CHECK(ParseProtoUnlimited(&proto, rdma_addr_, meta_data_->proto_size_))
        << "fail to parse proto from array";
    Status s = dst_dev_->MakeTensorFromProto(proto, recv_args_.alloc_attrs,
                                             result_tensor_);
    Done(s);
  }
}

void RdmaTensorRequest::RecvErrorStatus(const Status& status) {
  if (result_tensor_ == nullptr) {
    result_tensor_ = new Tensor();
  }
  LOG(ERROR) << "Received RDMA_MESSAGE_ERROR_STATUS: " << status.ToString();
  Done(status);
}

void RdmaTensorRequest::Start() {
  meta_data_ = channel_->GetTensorMetaData(key_);
  if (meta_data_ != nullptr) {
    AllocateTensorsAsync(
        [this](const Status& s) { Send(RDMA_MESSAGE_TENSOR_REQUEST); });
  } else {
    Send(RDMA_MESSAGE_TENSOR_REQUEST);
  }
}
}  // end namespace tensorflow

#endif
