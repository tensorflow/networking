#pragma once
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class Device;

namespace test {

// Provides a handle to a set of TensorFlow servers (masters and
// workers) for testing purposes.
//
// This class currently runs the servers in separate processes; the
// lifetime of this object is coterminous with the lifetimes of those
// processes.
class TestCluster {
 public:
  // Creates a new test cluster based on the given `options` (which
  // configure the number of devices of each type) and a count of
  // processes `n`. On success, the test cluster is stored in
  // *out_cluster, and this function returns OK. Otherwise an error is
  // returned.
  static Status MakeTestCluster(const SessionOptions& options, int n,
                                std::unique_ptr<TestCluster>* out_cluster);

  // As above, but allows overridding the server binary path via `binary_path`.
  static Status MakeTestCluster(const string& binary_path,
                                const SessionOptions& options, int n,
                                std::unique_ptr<TestCluster>* out_cluster);
  ~TestCluster();

  // Returns a vector of string "<hostname>:<port>" pairs that may be
  // used as targets to construct a GrpcSession.
  const std::vector<string>& targets() const { return targets_; }

  // Returns a vector of devices available in this test cluster.
  const std::vector<DeviceAttributes>& devices() const { return devices_; }

 private:
  TestCluster() = default;

  std::vector<std::unique_ptr<SubProcess>> subprocesses_;
  std::vector<string> targets_;
  std::vector<DeviceAttributes> devices_;

  TF_DISALLOW_COPY_AND_ASSIGN(TestCluster);
};

}  // end namespace test
}  // end namespace tensorflow
