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

#ifndef TENSORFLOW_CONTRIB_VERBS_RDMA_H_
#define TENSORFLOW_CONTRIB_VERBS_RDMA_H_

#ifdef TENSORFLOW_USE_VERBS

#include <infiniband/verbs.h>
#include <cstring>  // for memset
#include <functional>
#include <memory>  // for shared_ptr
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thread>
#include <deque>

#include "tensorflow_networking/verbs/verbs_util.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow_networking/verbs/verbs_service.pb.h"
#include "tensorflow_networking/verbs/grpc_verbs_client.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow_networking/verbs/rdma_mgr.h"


namespace tensorflow {
#define PKEY_DEFAULT 0
#define QUEUE_DEPTH_DEFAULT 1024
#define TIMEOUT_DEFAULT 14
#define RETRY_CNT_DEFAULT 7
#define SL_DEFAULT 0
#define TRAFFIC_CLASS 0

#define RDMA_LOG_0 LOG(INFO)
#define RDMA_LOG_1 VLOG(1)
#define RDMA_LOG_2 VLOG(2)
#define RDMA_LOG(LEVEL) RDMA_LOG_##LEVEL

struct RdmaParams {
  uint8_t port_num;
  uint8_t sgid_index;
  uint8_t pkey_index;
  uint32_t queue_depth;
  uint8_t timeout;
  uint8_t retry_cnt;
  uint8_t sl;
  enum ibv_mtu mtu;
  uint8_t traffic_class;
};
// structure to save the address of remote channels.
struct RdmaAddress {
  uint32_t lid;
  uint32_t qpn;
  uint32_t psn;
  uint64_t snp;
  uint64_t iid;
};
// structure to save information for remote memory regions.
struct RemoteMR {
  uint64_t remote_addr;
  uint32_t rkey;
};
enum BufferStatus { none, idle, busy };
enum Location { local, remote };

enum RdmaMessageType {
  RDMA_MESSAGE_META_DATA_UPDATE,
  RDMA_MESSAGE_TENSOR_RE_REQUEST,
  RDMA_MESSAGE_TENSOR_REQUEST,
  RDMA_MESSAGE_ERROR_STATUS,
  RDMA_MESSAGE_DRIVER_BEGIN
};

struct RdmaMessage {
  RdmaMessageType type_;
  uint16_t name_size_;
  string name_;
  int64 step_id_;
  uint32_t request_index_;
  union {
    uint64_t remote_addr_;
#ifdef RDMA_DATA_VALIDATION
    uint64_t checksum_;
#endif
  };
  uint32_t rkey_;
  bool is_dead_;
  DataType data_type_;
  TensorShape tensor_shape_;
  size_t tensor_bytes_;

  // int64 create_micros_;

  // uint32_t remote_bytes_addr_key_;
  // uint64_t remote_bytes_addr_;
  // For error status:
  Status status_;

  // (wuyongyu02) add the 'create_micros' for cat log
  // type|name_size|name|step_id|request_index|remote_addr/checksum|rkey|...
  //   1B|    2B   | 512|  8B   |     8B      |       8B           | 4B |...
  // ...|is_dead|data_type|tensor_shape|tensor_bytes|create_micros |...
  // ...|    1B |   XB    |    XB      |    8B      |  8B          |...
  // ...|remote_bytes_addr|       error_status            |
  // ...|8B               |   size - 4B, proto - XB       |
  static const size_t kNameCapacity = 512;
  static const size_t kTypeStartIndex = 0;
  static const size_t kNameSizeStartIndex = kTypeStartIndex + sizeof(type_);
  static const size_t kNameStartIndex =
      kNameSizeStartIndex + sizeof(name_size_);
  static const size_t kStepIdStartIndex = kNameStartIndex + kNameCapacity;
  static const size_t kRequestIndexStartIndex =
      kStepIdStartIndex + sizeof(step_id_);
  static const size_t kRemoteAddrStartIndex =
      kRequestIndexStartIndex + sizeof(request_index_);
  static const size_t kChecksumStartIndex = kRemoteAddrStartIndex;
  static const size_t kRkeyStartIndex =
      kRemoteAddrStartIndex + sizeof(remote_addr_);
  static const size_t kIsDeadStartIndex = kRkeyStartIndex + sizeof(rkey_);
  static const size_t kDataTypeStartIndex =
      kIsDeadStartIndex + sizeof(is_dead_);
  static const size_t kTensorShapeStartIndex =
      kDataTypeStartIndex + sizeof(data_type_);
  static const size_t kTensorBytesStartIndex =
      kTensorShapeStartIndex + sizeof(TensorShape);
  // static const size_t kCreateMicrosStartIndex =
  //     kTensorBytesStartIndex + sizeof(tensor_bytes_);

  // static const size_t kErrorStatusStartIndex =
  //     kCreateMicrosStartIndex + sizeof(create_micros_);
  static const size_t kErrorStatusStartIndex =
      kTensorBytesStartIndex + sizeof(tensor_bytes_);

  static const size_t kErrorStatusMaxSize = 4096;

  static const size_t kMessageTotalBytes = kErrorStatusStartIndex;
  static const size_t kRdmaMessageBufferSize =
      kMessageTotalBytes + kErrorStatusMaxSize;
  static string CreateMessage(const RdmaMessage& rm);
  static void ParseMessage(RdmaMessage& rm, void* buffer);
};

// Parse a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure where the parsed message will be saved
//   buffer: the place where the raw message is stored
// Returns:
//   None
struct FussionMessages {
  /* data */
  static const size_t kRdmaMaxMessagesNumber = 50;
  uint32_t message_numbers;
  uint32_t message_size[kRdmaMaxMessagesNumber];
  std::string messages[kRdmaMaxMessagesNumber];
  /* func */
  static string CreateFusionMessages(const std::vector<RdmaMessage>& rmv);
  static void ParseFussionMessages(std::vector<RdmaMessage>& rmv, void* buffer);

  /* index */
  static const size_t kMessageNumbersStartIndex = 0;
  static const size_t kMessageSizeStartIndex = kMessageNumbersStartIndex +
      sizeof(message_numbers);
  static const size_t KStringMessagesStartIndex = kMessageSizeStartIndex +
      sizeof(message_size);
  static const size_t kTotalFussionMessageSize = KStringMessagesStartIndex +
      kRdmaMaxMessagesNumber * RdmaMessage::kRdmaMessageBufferSize;
};

class FakeAllocator : public Allocator {
  public:
    FakeAllocator(void* buffer) : buffer_(buffer) {}
    string Name() override { return "fake_allocator"; }
    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
      //simply return the pre-allocated data
      return buffer_;
    }
    void DeallocateRaw(void* ptr) override {
      //TODO(wyy): does the real owner will free buffer_?
      // free(buffer_);
      // port::AlignedFree(buffer_);
    }
 
  private:
    //data should be 64 bytes aligned
    void* buffer_ = nullptr;
};

class RdmaChannel;
class ChannelRecordTensorMetaData;
class RdmaSendDriverMgr;

// Immediate types for RDMA write
const int Const_kNumMessageBuffers = 80;  // origin 80
enum RdmaImmDataType {
  RDMA_IMM_MAX_REQUEST_ID = 0xFFFFFFFF - 2 * Const_kNumMessageBuffers - 2,
  RDMA_IMM_DATA_ACK = 0xFFFFFFFF - Const_kNumMessageBuffers - 1,
  RDMA_IMM_DATA_MESSAGE = 0xFFFFFFFF,
  RDMA_IMM_MIN_SENDMGR_BASE = int(RDMA_IMM_MAX_REQUEST_ID/2 + 1),
};

// Write types for RDMA write-complete events
enum RdmaWriteIDType {
  RDMA_WRITE_ID_ACK,
  RDMA_WRITE_ID_MESSAGE,
  RDMA_WRITE_ID_TENSOR_WRITE,
  RDMA_WRITE_ID_SEND_DEIVER_WRITE
};

// Context for RDMA write-complete events
class RdmaWriteID {
 public:
  RdmaWriteID(RdmaWriteIDType write_type, void* write_context)
      : write_type(write_type), write_context(write_context) {}

  RdmaWriteIDType write_type;
  void* write_context;
};

// Tensor meta-data
class TensorMetaData {
 public:
  TensorShape tensor_shape_;
  DataType data_type_;
  size_t proto_size_;
  bool is_dead_;
  uint32 uid_;
  // record is the mata change for send-driven
  bool meta_changed_ = false;

  std::ostream& print(std::ostream& out) const {
    out << "Dtype = " << DataTypeString(data_type_)
        << ", Shape = " << tensor_shape_.DebugString() << ", Proto size = 0x"
        << std::hex << proto_size_ << ", Is dead = " << is_dead_;
    return out;
  }
};

inline std::ostream& operator<<(std::ostream& out,
                                const TensorMetaData& meta_data) {
  return meta_data.print(out);
}

void MRDeleter(ibv_mr* mr);
using MemoryRegionPtr = std::unique_ptr<ibv_mr, decltype(&MRDeleter)>;

// RdmaMemoryMgr
// Manages the local meta-data cache, and the registered RDMA memory regions.
class RdmaMemoryMgr {
 public:
  RdmaMemoryMgr(struct ibv_pd* pd) :pd_(pd) {}
  // static RdmaMemoryMgr& Singleton() {
  //   static RdmaMemoryMgr instance;
  //   return instance;
  // }

  ibv_mr* FindMemoryRegion(void* addr, size_t length);

  void InsertMemoryRegion(void* addr, size_t length,
                          const std::string& allocator_name);
  void EvictMemoryRegion(void* addr, size_t length);

  static bool Comparator(const void* ptr, const MemoryRegionPtr& other) {
    return ptr < reinterpret_cast<char*>(other->addr) + other->length;
  }

  struct ibv_pd* pd_;

 private:
  // Managed memory regions
  mutex mrs_mu_;
  std::vector<MemoryRegionPtr> mrs_ GUARDED_BY(mrs_mu_);
};

class RecordTensorMetaData {
 public:
  RecordTensorMetaData() {
    // stop_.store(true, std::memory_order_relaxed);
    total_bytes_ = 0;
  }

  ~RecordTensorMetaData() {
    // stop_.store(false, std::memory_order_relaxed);
  }

  static RecordTensorMetaData& Singleton() {
    static RecordTensorMetaData instance;
    return instance;
  }

  static uint32 GetTensorLength(const DataType& date_type,
                                const TensorShape& tensor_shape) {
    return GetEnumSize(date_type) * tensor_shape.num_elements();
  }

  static uint32 GetEnumSize(const DataType& date_type);

  void GlobalRecord(const std::string& origin_tensor_name,
                    const TensorMetaData& m, bool stop_record=false);

  typedef std::unordered_map<std::string, TensorMetaData> GTensorMetaType;
  typedef std::unordered_map<uint32, std::string> GTensorsUidKeyType;

  const GTensorMetaType& GetGlobalTensorsMetaData() {
    return global_tensors_meta_data_;
  }

  const GTensorsUidKeyType& GetGlobalTensorsUidParsedkey() {
    return global_tensors_uid_parsed_key_;
  }

  string DebugString() const;

  void WriteOutput(const std::string& content) const;

  void ReadFile(const std::string& filename, StringPiece* content);

 private:
  mutex global_tensor_meta_data_mu_;
  GTensorMetaType global_tensors_meta_data_;
  GTensorsUidKeyType global_tensors_uid_parsed_key_;
  // uid_ should less RDMA_IMM_MAX_REQUEST_ID
  uint32 uid_ = RDMA_IMM_MIN_SENDMGR_BASE;
  // std::atomic<bool> stop_;
  uint64 total_bytes_;
  string local_worker_name_ =  "";
};

// which is a member of RdmaChannel
class LocalDriverBufferMgr {
 public:
  explicit LocalDriverBufferMgr(RdmaChannel* channel) : channel_(channel) {
    DCHECK(channel != nullptr)
        << "LocalDriverBufferMgr construct channel is nullptr.";
  }

  typedef Rendezvous::DoneCallback DoneCallback;
  typedef Rendezvous::Args Args;
  typedef Rendezvous::ParsedKey ParsedKey;
  struct Item {
    mutex item_lock_;
    DoneCallback waiter = nullptr;
    Tensor* value;
    bool is_dead = false;
    bool has_value =  false;
    Args send_args;
    Args recv_args;
    CancellationToken cancellation_token;
    uint64 send_start_micros_;
    uint64 recv_start_micros_;
    uint64 request_start_micros_;

    ~Item() {
      if (send_args.device_context) {
        send_args.device_context->Unref();
      }
      if (recv_args.device_context) {
        recv_args.device_context->Unref();
      }
      if (value != nullptr) {
        // delete value;
      }
    }

    // Returns true iff this item represents a value being sent.
    bool HasCallback() const { return this->waiter != nullptr; }

    bool HasValue() const { return this->has_value;}
  };

  typedef std::deque<Item*> ItemQueue;

  struct QueueItems {
    ItemQueue* queue;
    mutex queue_lock_;
  };


  typedef gtl::FlatMap<string, Item*> Table;

  typedef gtl::FlatMap<string, QueueItems*> QueueTable;


  size_t InitLocalDriverBufferMgr();

  Status RdmaSave(const string& key, const Args& send_args, const Tensor& val,
                  const bool is_dead);

  Status QueueRdmaSave(const string& key, const Args& send_args,
                       Tensor* val, const bool is_dead,
                       const uint64& send_begin_micros);

  void LoadAsync(const string& key, const Args& recv_args,
                 DoneCallback done);

  void QueueLoadAsync(const string& key, const Args& recv_args,
                      DoneCallback done, const uint64& request_start_micros);

  void StartAbort(const Status& status);

  ~LocalDriverBufferMgr() {
    if (!table_.empty()) {
      StartAbort(errors::Cancelled("LocalDriverBufferMgr deleted"));
    }
  }

 public:
  bool use_queue_item_ = true;

 private:
  RdmaChannel* channel_;  // not owned
  Table table_;  // GUARDED_BY(mu_);
  QueueTable queue_table_;
  Status status_ = Status::OK();  // GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(LocalDriverBufferMgr);
};

class RemoteBytesAddrMemoryRegion;

// RdmaTensorRequest
// Represents a single tensor request.
class RdmaTensorRequest {
 public:
  typedef Rendezvous::DoneCallback RecvDoneCallback;

  // Creates a tensor request identified by index.
  RdmaTensorRequest(uint32_t index, const string& key, int64 step_id,
                    RdmaChannel* channel, Device* dst_dev,
                    const Rendezvous::Args recv_args,
                    const RecvDoneCallback& done);
  ~RdmaTensorRequest();

  // Request unique index.
  uint32_t index() { return index_; }

  // Start the tensor request sequence.
  //
  // 1. Allocate the result tensor (and proxy tensor if required).
  // 2. Send RDMA_MESSAGE_TENSOR_REQUEST to the remote side.
  void Start();

  // Receive tensor meta-data.
  //
  // 1. Update the local meta-data cache.
  // 2. Reallocate the result tensor (and proxy tensor if required).
  // 3. Re-send the request to the remote side.
  void RecvTensorMetaData(DataType dtype, TensorShape shape, bool is_dead,
                          size_t proto_size);

  // Receive tensor content (RDMA write was completed).
  //
  // Decode proto if required and/or move to GPU if the content was not
  // written to it directly (GPU direct is not available). Afterwards,
  // invoke Done().
  void RecvTensorContent();

  // Receive error status (in case of a remote error).
  // Invoke Done() with the status code.
  void RecvErrorStatus(const Status& status);

  RdmaChannel* rdma_channel() {
    return channel_;
  }

#ifdef RDMA_DATA_VALIDATION
  // Receive tensor checksum
  //
  // For validation: Get and store the Tensor's expected checksum for the
  // current request. Compare the result Tensor's checksum with the stored
  // checksum right before invoking Done().
  void RecvTensorChecksum(uint64_t checksum) { checksum_ = checksum; }
#endif
  uint64_t begin_start_req_;
  string key_;
  // SendMetaData message micros
  // uint64_t rm_create_micros_;
  RecvDoneCallback done_;
  Rendezvous::Args recv_args_;

 private:
  void Done(const Status& s);
  void Send(RdmaMessageType message_type);
  bool AllocateTensors();
  void AllocateTensorsAsync(StatusCallback done);
  void DeallocateTensors();

  size_t GetTensorLength(const TensorMetaData& meta);

  uint32_t index_;
  int64 step_id_;
  RdmaChannel* channel_;
  Device* dst_dev_;

  const TensorMetaData* meta_data_;
  FakeAllocator* fake_allocator_ = nullptr;
  Tensor* result_tensor_;

  std::shared_ptr<RemoteBytesAddrMemoryRegion> result_region_;
  Tensor* proxy_tensor_;
  void* rdma_addr_;
  // void* rdma_remote_bytes_addr_ = nullptr;
  // ibv_mr* remote_bytes_addr_mr_ = nullptr;
  ibv_mr* mr_;
#ifdef RDMA_DATA_VALIDATION
  uint64_t checksum_;
#endif
};

struct DriverEntry;

// RdmaTensorResponse
// Represents a single tensor response.
class RdmaTensorResponse {
 public:
  // Creates a response for request message.
  RdmaTensorResponse(RdmaChannel* channel, const RdmaMessage& rm)
      : channel_(channel), rm_(rm) {
    //   strings::StrCat(
    // src_device, ";", strings::Uint64ToHexString(src_incarnation, buf), ";",
    // dst_device, ";", name, ";", frame_iter.frame_id, ":",
    // frame_iter.iter_id);
    if (!rm.name_.empty()) {
      size_t found = rm.name_.find(";");
      string str = rm.name_.substr(found + 1, rm.name_.size());

      found = str.find(";");
      str = str.substr(found + 1, str.size());

      found = str.find(";");
      req_to_device_ = str.substr(0, found);
      parsed_key_ = rm.name_;
    } else {
      req_to_device_ = "";
      parsed_key_ = "";
    }
  }

  void Update(const RdmaMessage& rm) { rm_ = rm; }

  // Start the tensor response sequence.
  //
  // 1. Find the tensor in the local tag-match table and invoke RecvHandler.
  //    (Using RecvLocalAsync()).
  // 2. Compare the tensor's meta-data to the meta-data in the message (taken
  //    from the requester's local cache).
  //    If meta-data changed:
  //    a. Clone the tensor to be sent later.
  //    b. Send a meta-data update message and wait for re-request.
  //    Else:
  //    a. Send the tensor's content (using direct RDMA write).
  void Start();

  // Resume the response sequence, after a re-request.
  //
  // 1. Send the tensor's content that was cloned earlier.
  void Resume();

  // Destroy the response's resources and remove it from the pending list.
  void Destroy();

 public:
  uint64 request_index_;
  uint64 recv_local_send_rdma_;
  uint64 recv_send_content_ = 0;
  uint64 send_meta_begin_;
  string parsed_key_;
  string req_to_device_;

 private:
  void RecvHandler(const Rendezvous::Args& send_args,
                   const Rendezvous::Args& recv_args, const Tensor& in,
                   bool is_dead);
  void Clone(const Tensor& in, const TensorProto& proto, bool is_dead);


  void RdmaClone(const Tensor& in, const TensorProto& proto,
                 bool is_dead);

  void Send(const Tensor& in, const TensorProto& proto, bool is_dead,
            const Status& status);
  void SendBck(const Tensor& in, const TensorProto& proto, bool is_dead,
              const Status& status);

  bool TensorMetaDataChanged(const Tensor& in, bool is_dead);
  Status PrepareRecvTensor(const Rendezvous::ParsedKey& parsed,
                           Device** src_dev);
  void SendMetaData(const Tensor& in, const TensorProto& proto, bool is_dead);
  void SendContent(const Tensor& in, const TensorProto& proto, bool is_dead,
                   bool is_resume);
  void SendErrorStatus(const Status& status, const std::string& src_func_name);

  RdmaChannel* channel_;
  RdmaMessage rm_;  // The request message
  Device* src_dev_ = nullptr;
  TensorBuffer* src_buffer_ = nullptr;
  void* src_addr_ = nullptr;
  ibv_mr* mr_ = nullptr;
  uint64_t checksum_ = 0;
  bool meta_data_changed_ = false;

  // Re-item:
  TensorProto* proto_ = nullptr;
  Tensor* tensor_ = nullptr;
  bool is_dead_ = false;

  std::shared_ptr<RemoteBytesAddrMemoryRegion> res_region_;
  FakeAllocator* res_fake_allocator_;
};

class Chunk {
 public:
  Chunk(struct ibv_pd* pd);

  void FreeChunk();

  ~Chunk();

  void Alloc(size_t size, void** p, ibv_mr** mr, size_t realloc_size=0);

 private:
  void* new_p_;
  ibv_mr* new_mr_;
  size_t chunk_addr_size = 64*1024*1024;
  uint64 offset_;
  uint64 curr_size_;
  uint64 empty_size_;
  uint64 total_waste_size_;
  uint64 total_realloc_size_;
  mutex alloc_mu_;
  int allocate_size_;
  struct ibv_pd* pd_;
  std::vector<ibv_mr*> mrs_;
  std::vector<void*> chunk_addrs_;
};

class RdmaMessageBuffer;
// Class that represents the Rdma Adapter.
// Responsible for creation of the completion queue, and handling
// of work completions.
class RdmaAdapter {
  friend class RdmaChannel;
  friend class RdmaMessageBuffer;
  friend class RdmaTensorResponse;
  friend class RdmaTensorRequest;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;
  friend class RdmaSendDriverMgr;
  friend class ChannelRecordTensorMetaData;

 public:
  RdmaAdapter(const WorkerEnv* worker_env);
  ~RdmaAdapter();
  // Adapter name, e.g. mlx5_0.
  string name() const;
  void StartPolling();
  void Pool_Process_CQ(int cq_num);
  void Process_WR(ibv_wc wc_, int cq_num);

 protected:
  thread::ThreadPool* pool_;
  static const int MAX_CONCURRENT_WRITES = 5000;  // origin 1000 , second 5000
  ibv_context* context_;
  // RDMA configuration parameters
  RdmaParams params_;
  // ibverbs protection domain
  ibv_pd* pd_;
  // Completion event channel, to wait for work completions
  ibv_comp_channel** event_channel_vec_;

  // Completion queue, to poll on work completions
  ibv_cq** cq_vec_;
  //
  int cq_nums_;
  // Pre-allocated work completions array used for polling
  ibv_wc** wc_vec_;
  // worker env for thread
  const WorkerEnv* worker_env_;
  // thread for cq.
  std::vector<std::unique_ptr<Thread> > polling_thread_vec_;
  Chunk* recv_chunk_ = nullptr;
};

// Class that represents a connection to a remote Rdma peer.
// Responsible for connecting queue pairs.
class RemoteBytesAddrMemoryRegion {
 public:
  RemoteBytesAddrMemoryRegion(void* addr, ibv_mr* mr, size_t s) {
    mr_ptr_ = mr;
    addr_ = addr;
    size_ = s;
    ref_.store(0);
  }

  // TODO(wuyongyu02) need ibv_dereg_mr mr_ptr_
  ~RemoteBytesAddrMemoryRegion() {
    if (mr_ptr_!= nullptr && addr_ != nullptr) {
      // ibv_dereg_mr(mr_ptr_);
      // free(addr_);
      addr_ = nullptr;
      mr_ptr_ = nullptr;
    }
  } 

  bool RefCountIsOne() const {
    return (ref_.load(std::memory_order_acquire) >= 1);
  }

  void Ref() const {
    ref_.fetch_add(1, std::memory_order_relaxed);
  }

  bool Unref() const {
    ref_.store(0);
    return true;
  }

  mutable std::atomic_int_fast32_t ref_;
  void* addr_;
  ibv_mr* mr_ptr_;
  size_t size_;
};

// save bytes info
struct DriverPrefixMessage {
  TensorShape tensor_shape_;
  size_t tensor_bytes_;
  bool is_dead_;
  uint64 send_micros_;
  // for not meta changed
  static const size_t CKIsDeadIndexStartIndex = 0;
  static const size_t CkSendMiscrosStartIndex =
                                CKIsDeadIndexStartIndex + sizeof(is_dead_);
  static const size_t CkPrefixMessageTotalBytes =
                                CkSendMiscrosStartIndex + sizeof(send_micros_);

  static const size_t kTensorShapeStartIndex = 0;
  static const size_t kTensorBytesStartIndex =
                                kTensorShapeStartIndex + sizeof(tensor_shape_);
  static const size_t KIsDeadIndexStartIndex =
                                kTensorBytesStartIndex + sizeof(tensor_bytes_);

  static const size_t KSendMicrosStartIndex =
                                KIsDeadIndexStartIndex + sizeof(is_dead_);

  static const size_t kPrefixMessageTotalBytes =
                                KSendMicrosStartIndex + sizeof(send_micros_);

  static std::string CreateDriverPrefixMessage(const TensorShape& shape,
      const size_t& tensor_bytes, const bool& is_dead,
      const uint64& send_micros, const bool& meta_changed) {
    if (meta_changed) {
      char message[kPrefixMessageTotalBytes + 100];
      memcpy(message + kTensorShapeStartIndex, &shape, sizeof(shape));
      memcpy(message + kTensorBytesStartIndex, &tensor_bytes,
              sizeof(tensor_bytes));
      memcpy(message + KIsDeadIndexStartIndex, &is_dead, sizeof(is_dead));
      memcpy(message + KSendMicrosStartIndex, &send_micros,
              sizeof(send_micros));
      return std::string(message, kPrefixMessageTotalBytes);
    } else {
      char message[CkPrefixMessageTotalBytes + 100];
      memcpy(message + CKIsDeadIndexStartIndex, &is_dead, sizeof(is_dead));
      memcpy(message + CkSendMiscrosStartIndex, &send_micros,
              sizeof(send_micros));
      return std::string(message, CkPrefixMessageTotalBytes);
    }
  }

  static DriverPrefixMessage ParseDriverPrefixMessage(void* addr,
      const bool& meta_changed) {
    if (meta_changed) {
      char* message = static_cast<char*>(addr);
      DriverPrefixMessage m;
      memcpy(&m.tensor_shape_, message + kTensorShapeStartIndex,
              sizeof(m.tensor_shape_));
      memcpy(&m.tensor_bytes_, message + kTensorBytesStartIndex,
              sizeof(m.tensor_bytes_));
      memcpy(&m.is_dead_, message + KIsDeadIndexStartIndex,
              sizeof(m.is_dead_));
      memcpy(&m.send_micros_, message + KSendMicrosStartIndex,
              sizeof(m.send_micros_));
      return m;
    } else {
      char* message = static_cast<char*>(addr);
      DriverPrefixMessage m;
      memcpy(&m.is_dead_, message + CKIsDeadIndexStartIndex,
              sizeof(m.is_dead_));
      memcpy(&m.send_micros_, message + CkSendMiscrosStartIndex,
              sizeof(m.send_micros_));
      return m;
    }
  }
};

enum DriverStatus {
  DRIVER_INIT,
  RPC_0,
  RPC_1,
  DATA_NOT_READY,
  DATA_READY,
  DRIVER_ERROR
};
struct DriverEntry {
 public:
  DriverEntry(const uint32& uid,
              const std::string& parsedkey,
              void* addr,
              ibv_mr* mr,
              int allocate_size);

  DriverEntry();

  uint32 uinque_id_;
  std::string parsed_key_;
  std::atomic<DriverStatus> dri_status_;
  // saved tensor data and string message
  std::shared_ptr<RemoteBytesAddrMemoryRegion> mem_mr_;
  // uint32 prefix_msg_len_;
  std::string prefix_msg_;
  int allocate_size_ = 0;
  //
  uint32_t lkey_;
  //
  uint64_t addr_;
  // record metag changed
  bool meta_changed_ = false;


  // allocate for send prefix string
  std::shared_ptr<RemoteBytesAddrMemoryRegion> send_mem_mr_;

  // for send tensor ref
  TensorBuffer* src_buffer_ = nullptr;
  // for send tensor smr_
  struct ibv_mr* smr_ = nullptr;  // not owend
  // can memcpy tensor
  void* tensor_addr_ = nullptr;

  int local_allocate_size_ = 0;

  // allocate for send tensor
  std::shared_ptr<RemoteBytesAddrMemoryRegion> send_region_;

  uint64 send_micros_ = 0;
};

class RdmaSendDriverMgr {
 friend class RdmaChannel;
 friend class ChannelRecordTensorMetaData;
 friend class RdmaAdapter;

 public:
  RdmaSendDriverMgr(RdmaChannel* channel);

  size_t InitLocalDriverEntry();

  void NotifyRemoteDriverEntry();

  ~RdmaSendDriverMgr() {
  }

  // send service update recv_entries_
  void RpcUpdateRemoteDriverEntry(const DriverMessageReq* request,
                                 DriverMessageResp* response);

  // recv client update driver_entries_
  void RpcUpdateDriverEntries(const DriverMessageResp& resp);

  bool RpcReqResp(GrpcVerbsClient* client, const DriverMessageReq& req);

  void AllocateRecvEntriesStringMemoryAndRegion();

  std::shared_ptr<DriverEntry> GetRecvEntry(const std::string& parsed_key,
                                            bool* has_data);

  std::shared_ptr<DriverEntry> GetDriverEntry(const std::string& parsed_key,
                                              bool* has_data);

 public:
  std::atomic<bool> driver_mgr_is_ok_;
  typedef std::unordered_map<std::string,
                             std::shared_ptr<DriverEntry> > EntryMapType;
  // typedef absl::flat_hash_map<string,
  //                               std::shared_ptr<DriverEntry> > EntryMapType;

 protected:
  RdmaChannel * channel_;
  EntryMapType driver_entries_;
  EntryMapType recv_entries_;
};

class ChannelRecordTensorMetaData {
 public:
  // typedef absl::flat_hash_map<string, TensorMetaData> RecordMapType;
  typedef std::unordered_map<std::string, TensorMetaData> RecordMapType;
  typedef std::unordered_map<uint32, std::string> RecordMapUniIdType;

  ChannelRecordTensorMetaData(RdmaChannel* channel);

  static uint32 GetEnumSize(const DataType& date_type);

  static int GetTensorBytes(const TensorMetaData& m);

  void AllocateMemoryAndRegion(const string& key,
                               const TensorMetaData& m,
                               ibv_pd* pd,
                               void** addr,
                               ibv_mr** mr,
                               int* addr_size,
                               Allocator* alloc_attr = nullptr) const;

  void AllocateSendStringMemoryAndRegion(ibv_pd* pd,
                                         void** addr,
                                         ibv_mr** mr,
                                         int* addr_size,
                                         Allocator* alloc_attr = nullptr);

  void Record(const std::string& tensor_name,
              const TensorMetaData& m);

  static StringPiece ConsumeNextPart(StringPiece* s, char delim);

  static string RegexEdgeName(const string & str);

  void InitMetaDataFromEnv();

  const RecordMapType & GetChannelTensorsMetaData() {
    return channel_tensors_meta_data_;
  }

  const RecordMapUniIdType & GetChannelTensorsUidParsedkey() {
    return channel_tensors_uid_parsed_key_;
  }

 public:
  RecordMapType channel_tensors_meta_data_;

  RecordMapUniIdType channel_tensors_uid_parsed_key_;

 private:
  RdmaChannel* channel_;
  mutex channel_tensor_meta_data_mu_;
  // uid_ must less RDMA_IMM_MAX_REQUEST_ID
  uint32 uid_ = RDMA_IMM_MIN_SENDMGR_BASE;
};

class RdmaMgr;
class RdmaChannel {
  friend class RdmaAdapter;
  friend class RdmaMessageBuffer;
  friend class RdmaTensorBuffer;
  friend class RdmaTensorRequest;
  friend class RdmaTensorResponse;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;
  friend class RdmaSendDriverMgr;
  friend class ChannelRecordTensorMetaData;

 public:
  explicit RdmaChannel(const RdmaAdapter* adapter, const string local_name,
                       const string remote_name_, GrpcChannelCache* rdma_mgr,
                       ibv_cq* cq);

  ~RdmaChannel();
  inline const RdmaAddress& self() { return self_; }
  RdmaAddress address() const;
  inline const std::vector<RdmaMessageBuffer*>& message_buffers() const {
    return message_buffers_;
  }
  void Connect(const RdmaAddress& remoteAddr);
  void Connect();
  void Recv();
  void SetRemoteAddress(const RdmaAddress& ra, bool override);

  // Requests:
  RdmaTensorRequest* InsertTensorRequest(
      const string& key, int64 step_id, Device* dst_dev,
      const Rendezvous::Args recv_args,
      const RdmaTensorRequest::RecvDoneCallback& done);
  void RemoveTensorRequest(uint32_t request_index);
  RdmaTensorRequest* GetTensorRequest(uint32_t request_index);

  // Responses:
  RdmaTensorResponse* AddTensorResponse(const RdmaMessage& rm);
  RdmaTensorResponse* UpdateTensorResponse(const RdmaMessage& rm);
  void RemoveTensorResponse(uint32_t request_index);

  // static const int kNumMessageBuffers = 2;
  static const int kNumMessageBuffers = Const_kNumMessageBuffers;
  static const int kPingRecvWrid = 0;
  // CAT log
  RdmaTensorRequest* GetTensorRequestForCat(uint32_t request_index);

  inline size_t Alloc(size_t size, void** p, ibv_mr** mr,
                      bool dynamic=false, size_t realloc_size=0) const;
  bool FindLocalMr(const std::string& key, void** remote_bytes_addr,
                   ibv_mr** mr, int* length);

  inline void FindOrCreateRemoteBytesAddrMemoryRegion(const std::string& key,
      void** remote_bytes_addr /*new*/,
      ibv_mr** mr /*new*/,
      std::shared_ptr<RemoteBytesAddrMemoryRegion> * region,
      size_t length,
      const Allocator* alloc_attr = nullptr);

  size_t ChannelAllocateTensors(const string& key, const TensorMetaData& meta,
      const Allocator* alloc_attr,   ibv_mr** mr/*new*/,
      std::shared_ptr<RemoteBytesAddrMemoryRegion> * region,
      void** rdma_addr /*new*/);

  GrpcChannelCache* GetChannelChache() { return channel_cache_; }

  std::shared_ptr<RdmaSendDriverMgr> GetRdmaSendDriverMgr() {
    return rdma_send_driver_mgr_;
  }

  // For tensor response

  // For Send Kernel op
  void SendDriverData(const Tensor& in,
                      bool is_dead,
                      const std::string& name);

  // (1) enter
  void InitAndSetDriverStatus();

  void TestPleSendOrCheck() {
    LOG(INFO) << "TestPleSendOrCheck begin...";
    PleSendOrCheck();
  }

  void FakeAllocateTest() {
    Tensor fill_shape_tensor(DT_INT32, TensorShape({1}));
    fill_shape_tensor.vec<int32>()(0) = 1;
    // fill_shape_tensor.vec<int32>()(1) = 256;
    // fill_shape_tensor.vec<int32>()(2) = 1024;
    // fill_shape_tensor.vec<int32>()(3) = 1024;
    auto flat = fill_shape_tensor.flat<int32>();
    auto ts = fill_shape_tensor.scalar<int32>();
    LOG(INFO) << "ts size:" << ts.size() 
              << " flat size:" << flat.size();
    for (int i = 0; i < flat.size(); ++i) {
      // flat(i) = i;
      LOG(INFO) << "ts " << i << " :" << ts(i);
    }
  }

  void PleSendOrCheck();

  const TensorMetaData* GetTensorMetaData(const std::string& tensor_name);

  const TensorMetaData* SetTensorMetaData(const std::string& tensor_name,
                                          DataType dtype,
                                          const TensorShape& shape,
                                          bool is_dead, size_t proto_size);
  // Memory regions
  ibv_mr* FindMemoryRegion(void* addr, size_t length);

 public:
  bool could_send_driver_ = false;
  string local_name_;
  string remote_name_;
  std::shared_ptr<ChannelRecordTensorMetaData> channel_record_;
  std::shared_ptr<RdmaSendDriverMgr> rdma_send_driver_mgr_;
  std::shared_ptr<LocalDriverBufferMgr> local_driver_buffer_mgr_;

 private:
  static const int kPingBuffSize = 1024;
  char ping_buff_[kPingBuffSize];
  struct ibv_mr* mr_;
  struct ibv_sge ping_sge_list_;
  int PingPostRecv();
  int PingPostSend();

 protected:
  const RdmaAdapter* adapter_;
  RdmaMgr* rdma_mgr_;
  RdmaAddress self_;
  ibv_qp* qp_;
  mutex mu_;
  bool connected_ GUARDED_BY(mu_) = false;
  RdmaAddress remote_ GUARDED_BY(mu_);
  bool remote_set_ GUARDED_BY(mu_) = false;
  mutex ct_mu_;
  typedef std::unordered_map<uint32_t, RdmaTensorRequest> RequestTable;
  RequestTable request_table_ GUARDED_BY(ct_mu_);
  typedef std::unordered_map<string, uint32_t> ParsedKeyToIndex;
  typedef std::unordered_map<uint32_t, string> IndexToParsedKey;

  IndexToParsedKey req_table_idx_to_pkey_ GUARDED_BY(ct_mu_);

  uint32_t request_serial_ GUARDED_BY(ct_mu_);
  mutex responses_mu_;
  typedef std::unordered_map<uint32_t,
                         std::shared_ptr<RdmaTensorResponse> > ResponsesTable;
  ResponsesTable responses_table_ GUARDED_BY(responses_mu_);
  std::vector<RdmaMessageBuffer*> message_buffers_;
  // for addr size
  // Managed memory regions
  mutex remote_bytes_addr_mu_;
  typedef absl::flat_hash_map<string,
                     std::shared_ptr<RemoteBytesAddrMemoryRegion>> MRegionType;
  // typedef std::unordered_map<std::string,
  //   std::shared_ptr<RemoteBytesAddrMemoryRegion>> MRegionType;
  MRegionType remote_bytes_addr_mrs_ GUARDED_BY(remote_bytes_addr_mu_);

  GrpcChannelCache* const channel_cache_;

  // meta record
  mutex tensor_meta_data_mu_;
  std::unordered_map<std::string, TensorMetaData> tensors_meta_data_;

  // for mem allocator
  Allocator* rdma_mem_allocator_;
  RdmaMemoryMgr* rdma_memory_mgr_;
  std::vector<SubAllocator::Visitor> alloc_visitors_;
  std::vector<SubAllocator::Visitor> free_visitors_;
  struct ibv_pd *pd_;  // not owned
  size_t pagesize_ = sysconf(_SC_PAGESIZE);
};

// Class that represents a buffer for Rdma message sending.
class RdmaMessageBuffer {
  friend class RdmaChannel;
  friend class RdmaAdapter;
  friend class RdmaMgr;
  friend class RdmaRemoteRendezvous;

 public:
  explicit RdmaMessageBuffer(RdmaChannel* channel, string name);
  ~RdmaMessageBuffer();

  inline void* buffer() const { return buffer_; }
  inline ibv_mr* self() const { return self_; }
  inline void SetBufferStatus(Location loc, BufferStatus status) {
    mu_.lock();
    if (loc == local) {
      local_status_ = status;
    } else {
      remote_status_ = status;
    }
    mu_.unlock();
  }
  void FreeBuffer();
  void EnqueueItem(string Item);
  void SendNextItem();
  void CreateCPUBuffer(size_t size, bool lock = true);
  void ChunkCreateCPUBuffer(size_t size, void* buffer, ibv_mr* mr,
                            bool lock = true);
  void SetRemoteMR(RemoteMR rmi, bool override);
  void Write(uint32_t imm_data, size_t buffer_size);

  static void Write(const RdmaChannel* channel, uint32_t imm_data,
                    size_t buffer_size, uint64_t src_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey,
                    RdmaWriteIDType write_type, void* write_context);

  static void WriteWithPrefix(const RdmaChannel* channel, uint32_t imm_data,
                              size_t buffer_size, uint64_t src_addr,
                              uint32_t lkey, uint64_t remote_addr,
                              uint32_t rkey, RdmaWriteIDType write_type,
                              void* write_context, uint64_t prefix_addr,
                              uint32_t prefix_lkey, size_t prefix_size);

  static void SendAck(const RdmaChannel* channel, int pair_index);

 public:
  int pair_index_;
  uint64_t rm_ack_micros_;

 protected:
  int64 time_guard_;
  const RdmaChannel* channel_;
  void* buffer_ = nullptr;
  bool buffer_on_host_ = true;
  size_t size_ = 0;
  const string name_;
  ibv_mr* self_ = nullptr;
  mutex mu_;
  RemoteMR remote_;
  std::queue<string> queue_ GUARDED_BY(mu_);
  BufferStatus local_status_ GUARDED_BY(mu_) = none;
  BufferStatus remote_status_ GUARDED_BY(mu_) = none;
};

class VerbsEnvRegistrar {
 public:
  static VerbsEnvRegistrar* Instance() {
    static VerbsEnvRegistrar* instance_ = new VerbsEnvRegistrar();
    return instance_;
  }
  int RdmaCQpoolSize() {
    return rdma_cqpool_size_;
  }

  bool RdmaEnableSendDriven() {
    return enable_send_driven_;
  }

  int RdmaTensorBufferRatio() {
    return rdma_tensor_buffer_ratio_;
  }

  int RdmaCqNums() {
    return rdma_cq_nums_;
  }
  int RdmaChunkSize() {
    return rdma_chunk_size_;
  }

 private:
  VerbsEnvRegistrar() {
    rdma_cqpool_size_ = RDMACQPOOLSIZE();
    CHECK(rdma_cqpool_size_ < 500 && rdma_cqpool_size_ >= 1)
          << "rdma_cqpool_size_ must less 100 and greater 1";
    enable_send_driven_ = RDMAENABLESENDDRIERN() == 1 ? true : false;

    rdma_tensor_buffer_ratio_ = RDMATENSORBUFFERRATIO();
    CHECK(rdma_tensor_buffer_ratio_ < 100 && rdma_tensor_buffer_ratio_ >= 1)
          << "rdma_tensor_buffer_ratio_ must less 100 and greater 1";

    rdma_cq_nums_ = RDMACQNUMS();
    CHECK(rdma_cq_nums_ < 100 && rdma_cq_nums_ >= 1)
          << "rdma_cq_nums_ must less 100 and greater 1";
    rdma_chunk_size_ = RDMACHUNKSIZE();
  }

  int rdma_cqpool_size_;
  bool enable_send_driven_;
  int rdma_tensor_buffer_ratio_;
  int rdma_cq_nums_;
  int rdma_chunk_size_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // TENSORFLOW_CONTRIB_VERBS_RDMA_H_
