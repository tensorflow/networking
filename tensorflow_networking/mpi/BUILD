# Description:
#   MPI based communication interfaces and implementations for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# For platform specific build config
load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//third_party/mpi:mpi.bzl", "mpi_copts")
load("//third_party/mpi:mpi.bzl", "mpi_linkopts")

tf_proto_library_cc(
    name = "mpi_msg_proto",
    srcs = ["mpi_msg.proto"],
    cc_api_version = 2,
    protodeps = ["@org_tensorflow//tensorflow/core:worker_proto"],
    visibility = [
        "//tensorflow_networking:__subpackages__",
    ],
)

cc_library(
    name = "mpi_utils",
    srcs = ["mpi_utils.cc"],
    hdrs = ["mpi_utils.h"],
    copts = mpi_copts(),
    linkopts = mpi_linkopts(),
    deps = [
        "@org_tensorflow//tensorflow/core:core",
    ],
)

cc_library(
    name = "mpi_rendezvous_mgr",
    srcs = ["mpi_rendezvous_mgr.cc"],
    hdrs = ["mpi_rendezvous_mgr.h"],
    copts = mpi_copts(),
    linkopts = mpi_linkopts(),
    deps = [
        ":mpi_msg_proto_cc",
        ":mpi_utils",
        "@org_tensorflow//tensorflow/core:core",
        "@org_tensorflow//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime:recent_request_ids",
        "@org_tensorflow//tensorflow/core/distributed_runtime:request_id",
        "@org_tensorflow//tensorflow/core/distributed_runtime:session_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime:tensor_coding",
        "@org_tensorflow//tensorflow/core/distributed_runtime:worker_env",
    ],
)

cc_library(
    name = "mpi_server_lib",
    srcs = ["mpi_server_lib.cc"],
    hdrs = ["mpi_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":mpi_rendezvous_mgr",
        ":mpi_utils",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)
