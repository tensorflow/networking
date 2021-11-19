# Description:
#   Verbs RDMA communication interfaces and implementations for TensorFlow.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_library")

# For platform specific build config
load(
    "@org_tensorflow//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library_cc",
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_cuda_library",
)

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")

# For platform specific build config
load(
    "@org_tensorflow//tensorflow/core/platform:default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

load(
    "@org_tensorflow//tensorflow/core/platform:default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [
        "//tensorflow_networking:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

tf_proto_library_cc(
    name = "verbs_service_proto",
    srcs = ["verbs_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    visibility = [
        "//tensorflow_networking:__subpackages__",
    ],
)

cc_library(
    name = "verbs_util",
    srcs = ["verbs_util.cc"],
    hdrs = ["verbs_util.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:with_verbs_support": ["-libverbs"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "grpc_verbs_service",
    srcs = ["grpc_verbs_service.cc"],
    hdrs = ["grpc_verbs_service.h"],
    deps = [
        ":grpc_verbs_service_impl",
        ":rdma_mgr",
        ":rdma",
        ":verbs_service_proto_cc",
        "@org_tensorflow//tensorflow:grpc++",
        #"@org_tensorflow//tensorflow/core:lib_internal",
        "@org_tensorflow//tensorflow/core/distributed_runtime:session_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_verbs_service_impl",
    srcs = ["grpc_verbs_service_impl.cc"],
    hdrs = ["grpc_verbs_service_impl.h"],
    deps = [
        ":verbs_service_proto_cc",
        "@org_tensorflow//tensorflow:grpc++",
    ],
)

cc_library(
    name = "grpc_verbs_client",
    srcs = ["grpc_verbs_client.cc"],
    hdrs = ["grpc_verbs_client.h"],
    copts = ["-Og", "-g3"],
    deps = [
        ":grpc_verbs_service_impl",
        ":verbs_service_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime:call_options",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rdma_rendezvous_mgr",
    srcs = ["rdma_rendezvous_mgr.cc"],
    hdrs = ["rdma_rendezvous_mgr.h", "rdma.h"],
    copts = ["-Og", "-g3"],
    deps = [
        ":rdma_mgr",
        ":verbs_util",
        #"@org_tensorflow//tensorflow/core:core_cpu_internal",
        #"@org_tensorflow//tensorflow/core:gpu_runtime",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime:worker_env",
        #"@org_tensorflow//tensorflow/core/distributed_runtime:worker_cache_partial",
    ],
)

tf_cuda_library(
    name = "rdma_mgr",
    srcs = ["rdma_mgr.cc"],
    hdrs = ["rdma_mgr.h", "rdma.h"],
    copts = ["-Og", "-g3"],
    deps = [
        ":grpc_verbs_client",
        #":rdma",
        ":verbs_util",
        ":verbs_service_proto_cc",
        "@org_tensorflow//tensorflow/core",
        #"@org_tensorflow//tensorflow/core:core_cpu_internal",
        "@org_tensorflow//tensorflow/core:lib",
        #"@org_tensorflow//tensorflow/core:lib_internal",
        "@org_tensorflow//tensorflow/core/distributed_runtime:session_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime:worker_env",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
    ],
)


tf_cuda_library(
    name = "rdma",
    srcs = ["rdma.cc"],
    hdrs = ["rdma.h"],
    linkopts = select({
        "@org_tensorflow//tensorflow:with_verbs_support": ["-libverbs"],
        "//conditions:default": [],
    }),
    copts = ["-Og", "-g3"],
    deps = [
        ":rdma_mgr",
        ":grpc_verbs_client",
        ":verbs_service_proto_cc",
        ":verbs_util",
        #"@org_tensorflow//tensorflow/core:core_cpu_internal",
        "@org_tensorflow//tensorflow/core:framework",
        #"@org_tensorflow//tensorflow/core:framework_internal",
        #"@org_tensorflow//tensorflow/core:gpu_runtime",
        "@org_tensorflow//tensorflow/core:lib",
        #"@org_tensorflow//tensorflow/core:lib_internal",
        "@org_tensorflow//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "@org_tensorflow//tensorflow/core/distributed_runtime:session_mgr",
        "@org_tensorflow//tensorflow/core/distributed_runtime:worker_env",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_channel",
    ],
)

cc_library(
    name = "verbs_server_lib",
    srcs = ["verbs_server_lib.cc"],
    hdrs = ["verbs_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":grpc_verbs_service",
        ":rdma_mgr",
        ":rdma_rendezvous_mgr",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime:server_lib",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)

