package(default_visibility = ["//visibility:public"])

cc_library(
    name = "grpc_header_lib",
    hdrs = [":grpc_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "farmhash_header_lib",
    hdrs = [":farmhash_header_include"],
    includes = ["src"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    deps = [
        ":grpc_header_lib",
        ":farmhash_header_lib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_pywrap_tensorflow_internal",
    srcs = [":_pywrap_tensorflow_internal.so"],
    deps = [":libtensorflow_framework"],
    visibility = ["//visibility:public"],
)

%{FARMHASH_HEADER_GENRULE}
%{GRPC_HEADER_GENRULE}
%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
%{TF_PYWRAP_INTERNAL_LIBRARY_GENRULE}
