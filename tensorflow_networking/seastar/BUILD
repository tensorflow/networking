package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

COMMON_COPTS = [
    "-DFMT_HEADER_ONLY",
    "-DNO_EXCEPTION_HACK",
    "-DNO_EXCEPTION_INTERCEPT",
    "-DHAVE_LZ4_COMPRESS_DEFAULT",
]

cc_library(
    name = "seastar_tensor_coding",
    srcs = [
        "seastar_message.cc",
        "seastar_tensor_coding.cc",
    ],
    hdrs = [
        "seastar_message.h",
        "seastar_tensor_coding.h",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "seastar_worker_service",
    srcs = [
        "seastar_client_tag.cc",
        "seastar_server_tag.cc",
        "seastar_tag_factory.cc",
        "seastar_worker_service.cc",
    ],
    hdrs = [
        "seastar_client_tag.h",
        "seastar_server_tag.h",
        "seastar_tag_factory.h",
        "seastar_worker_interface.h",
        "seastar_worker_service.h",
        "seastar_worker_service_method.h",
    ],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_tensor_coding",
        "@local_config_tf//:tf_header_lib",
        "@seastar",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_cpuset",
    srcs = ["seastar_cpuset.cc"],
    hdrs = ["seastar_cpuset.h"],
    linkstatic = 1,
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_engine",
    srcs = [
        "seastar_client.cc",
        "seastar_engine.cc",
    ],
    hdrs = [
        "seastar_client.h",
        "seastar_engine.h",
    ],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_cpuset",
        ":seastar_worker_service",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_remote_worker",
    srcs = ["seastar_remote_worker.cc"],
    hdrs = ["seastar_remote_worker.h"],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_worker_service",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_worker_cache",
    srcs = [
        "seastar_channel_cache.cc",
        "seastar_worker_cache.cc",
    ],
    hdrs = [
        "seastar_channel_cache.h",
        "seastar_worker_cache.h",
    ],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_engine",
        ":seastar_remote_worker",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_rendezvous_mgr",
    srcs = ["seastar_rendezvous_mgr.cc"],
    hdrs = ["seastar_rendezvous_mgr.h"],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_worker_cache",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "seastar_server_lib",
    srcs = ["seastar_server_lib.cc"],
    hdrs = ["seastar_server_lib.h"],
    copts = COMMON_COPTS,
    linkstatic = 1,
    deps = [
        ":seastar_rendezvous_mgr",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
