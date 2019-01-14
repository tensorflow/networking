# Ops that communicate with other processes via MPI.

#package(default_visibility = [
#    "//tensorflow:__subpackages__",
#])

licenses(["notice"])  # Apache 2.0

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)


load("//third_party/mpi:mpi.bzl", "mpi_defines")
load("//third_party/mpi:mpi.bzl", "mpi_copts")
load("//third_party/mpi:mpi.bzl", "mpi_linkopts")

tf_proto_library_cc(
    name = "mpi_message_proto",
    srcs = ["mpi_message.proto"],
    cc_api_version = 2,
    protodeps = ["@org_tensorflow//tensorflow/core:protos_all"],
    visibility = [
        "//tensorflow_networking:__subpackages__",
    ],
)

cc_library(
    name = "mpi_defines",
    defines = mpi_defines(),
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_custom_op_py_library",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_py_test",
)

tf_custom_op_library(
    name = "python/ops/_mpi_ops.so",
    srcs = [
        "kernels/mpi_ops.cc",
        "kernels/ring.cc",
        "kernels/ring.h",
        "ops/mpi_ops.cc",
    ],
    gpu_srcs = [
        "kernels/ring.cu.cc",
        "kernels/ring.h",
    ],
    deps = [
        ":mpi_defines",
        ":mpi_message_proto_cc",
        "@org_tensorflow//tensorflow/core:stream_executor_headers_lib",
    ],
    copts = mpi_copts(),
    linkopts = mpi_linkopts(),
    defines = mpi_defines(),
)

tf_kernel_library(
    name = "mpi_ops_kernels",
    srcs = [
        "kernels/mpi_ops.cc",
        "kernels/ring.cc",
    ],
    hdrs = [
        "kernels/ring.h",
    ],
    gpu_srcs = [
        "kernels/ring.cu.cc",
    ],
    deps = [
        ":mpi_defines",
        ":mpi_message_proto_cc",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:gpu_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
#        "@org_tensorflow//tensorflow/core:proto_text",
#        "@org_tensorflow//tensorflow/core:stream_executor",
        "@org_tensorflow//tensorflow/core:gpu_lib",
    ],
    copts = mpi_copts(),
    linkopts = mpi_linkopts(),
    defines = mpi_defines(),
    # TODO: Include?    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["mpi_ops"],
)

tf_gen_op_wrapper_py(
    name = "mpi_ops",
    deps = [":mpi_ops_op_lib"],
)

tf_custom_op_py_library(
    name = "mpi_collectives_py",
    srcs = [
        "__init__.py",
        "python/ops/mpi_ops.py",
    ],
    dso = [
        ":python/ops/_mpi_ops.so",
    ],
    kernels = [
        ":mpi_ops_kernels",
        ":mpi_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":mpi_ops",
#        "@org_tensorflow//tensorflow/contrib/util:util_py",
#        "@org_tensorflow//tensorflow/python:device",
#        "@org_tensorflow//tensorflow/python:framework_ops",
        "@org_tensorflow//tensorflow/python:platform",
#        "@org_tensorflow//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "mpi_ops_test",
    srcs = ["mpi_ops_test.py"],
    additional_deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python:platform",
    ],
    data = [
        ":python/ops/_mpi_ops.so",
    ],
    tags = ["manual"],
)
