licenses(["permissive"])  # LGPL headers only

load("@//third_party:common.bzl", "template_rule")

exports_files(["LICENSE"])

cc_library(
    name = "gnutls",
    hdrs = glob([
        "lib/includes/gnutls/*.h",
    ]) + [
        "lib/includes/gnutls/gnutls.h",
    ],
    strip_include_prefix = "lib/includes",
    visibility = ["//visibility:public"],
)

template_rule(
    name = "gnutls_h",
    src = "lib/includes/gnutls/gnutls.h.in",
    out = "lib/includes/gnutls/gnutls.h",
    substitutions = {
        "#define GNUTLS_VERSION \"@VERSION@\"": "#define GNUTLS_VERSION \"3.6.12\"",
        "#define GNUTLS_VERSION_MAJOR @MAJOR_VERSION@": "#define GNUTLS_VERSION_MAJOR 3",
        "#define GNUTLS_VERSION_MINOR @MINOR_VERSION@": "#define GNUTLS_VERSION_MINOR 6",
        "#define GNUTLS_VERSION_PATCH @PATCH_VERSION@": "#define GNUTLS_VERSION_PATCH 12",
        "#define GNUTLS_VERSION_NUMBER @NUMBER_VERSION@": "#define GNUTLS_VERSION_NUMBER 0x03060c",
        "@DEFINE_IOVEC_T@": "#include <sys/uio.h>\ntypedef struct iovec giovec_t;",
    },
)
