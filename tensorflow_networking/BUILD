config_setting(
    name = "mpi_library_is_openmpi_based",
    values = {"define": "mpi_library_is_openmpi_based=true"},
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libtensorflow_networking.so",
    linkopts = ["-z defs"],
    linkshared = 1,
    deps = [
        "//tensorflow_networking/seastar:seastar_server_lib",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)
