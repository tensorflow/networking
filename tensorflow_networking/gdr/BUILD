# Description:
#   GPU Direct RDMA Out-of-Band Tensor transport for TensorFlow.

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
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

proto_library(
    name = "gdr_proto",
    srcs = ["gdr.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_proto_library(
    name = "gdr_proto_cc",
    deps = [":gdr_proto"],
)

cc_library(
    name = "gdr_memory_manager",
    srcs = ["gdr_memory_manager.cc"],
    hdrs = ["gdr_memory_manager.h"],
    linkopts = [
        "-libverbs",
        "-lrdmacm",
    ],
    deps = [
        ":gdr_proto_cc",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gdr_worker",
    srcs = ["gdr_worker.cc"],
    hdrs = ["gdr_worker.h"],
    deps = [
        ":gdr_memory_manager",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gdr_rendezvous_mgr",
    srcs = ["gdr_rendezvous_mgr.cc"],
    hdrs = ["gdr_rendezvous_mgr.h"],
    deps = [
        ":gdr_memory_manager",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gdr_collective_executor_mgr",
    srcs = ["gdr_collective_executor_mgr.cc"],
    hdrs = ["gdr_collective_executor_mgr.h"],
    deps = [
        ":gdr_memory_manager",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gdr_server_lib",
    srcs = ["gdr_server_lib.cc"],
    hdrs = ["gdr_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    visibility = [
        "//tensorflow_networking:__subpackages__",
    ],
    deps = [
        ":gdr_collective_executor_mgr",
        ":gdr_memory_manager",
        ":gdr_rendezvous_mgr",
        ":gdr_worker",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
