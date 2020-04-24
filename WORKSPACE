workspace(name = "tf_networking")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")
load("//tensorflow_networking:repo.bzl", "tensorflow_http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

tf_configure(
    name = "local_config_tf",
)

maybe(
    http_archive,
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

maybe(
    http_archive,
    name = "com_google_protobuf",
    sha256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59",
    strip_prefix = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

maybe(
    http_archive,
    name = "com_github_nelhage_rules_boost",
    sha256 = "f8c9653c1c49489c04f9f87ab1ee93d7b59bb26a39d9e30e9687fca3c6197c3f",
    strip_prefix = "rules_boost-9f9fb8b2f0213989247c9d5c0e814a8451d18d7f",
    urls = ["https://github.com/nelhage/rules_boost/archive/9f9fb8b2f0213989247c9d5c0e814a8451d18d7f.tar.gz"],
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

maybe(
    http_archive,
    name = "fmtlib",
    build_file = "//third_party:fmtlib.BUILD",
    sha256 = "3c812a18e9f72a88631ab4732a97ce9ef5bcbefb3235e9fd465f059ba204359b",
    strip_prefix = "fmt-5.2.1",
    urls = [
        "https://github.com/fmtlib/fmt/archive/5.2.1.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "cares",
    build_file = "//third_party/cares:cares.BUILD",
    sha256 = "03f708f1b14a26ab26c38abd51137640cb444d3ec72380b21b20f1a8d2861da7",
    strip_prefix = "c-ares-1.13.0",
    urls = [
        "https://c-ares.haxx.se/download/c-ares-1.13.0.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "yaml-cpp",
    build_file = "//third_party:yaml-cpp.BUILD",
    sha256 = "77ea1b90b3718aa0c324207cb29418f5bced2354c2e483a9523d98c3460af1ed",
    strip_prefix = "yaml-cpp-yaml-cpp-0.6.3",
    urls = [
        "https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.3.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "colm",
    build_file = "//third_party:colm.BUILD",
    sha256 = "4644956dd82bedf3795bb1a6fdf9ee8bdd33bd1e7769ef81ffdaa3da70c5a349",
    strip_prefix = "colm-0.13.0.6",
    urls = [
        "http://www.colm.net/files/colm/colm-0.13.0.6.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "ragel",
    build_file = "//third_party:ragel.BUILD",
    sha256 = "08bac6ff8ea9ee7bdd703373fe4d39274c87fecf7ae594774dfdc4f4dd4a5340",
    strip_prefix = "ragel-7.0.0.11",
    urls = [
        "http://www.colm.net/files/ragel/ragel-7.0.0.11.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "cryptopp",
    build_file = "//third_party:cryptopp.BUILD",
    sha256 = "e3bcd48a62739ad179ad8064b523346abb53767bcbefc01fe37303412292343e",
    strip_prefix = "cryptopp-CRYPTOPP_8_2_0",
    urls = [
        "https://github.com/weidai11/cryptopp/archive/CRYPTOPP_8_2_0.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "lz4",
    build_file = "//third_party:lz4.BUILD",
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2/lib",
    urls = [
        "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz",
    ],
)

maybe(
    tensorflow_http_archive,
    name = "seastar",
    build_file = "//third_party:seastar.BUILD",
    patch_file = "//third_party:seastar.patch",
    sha256 = "27f1d42e77acfb8bcccd102e417fdaa81b3c8d589a8e7b009dd3312dcf6fbeef",
    strip_prefix = "seastar-seastar-19.06.0",
    urls = [
        "https://github.com/scylladb/seastar/archive/seastar-19.06.0.tar.gz",
    ],
)

maybe(
    http_archive,
    name = "readerwriterqueue",
    build_file = "//third_party:readerwriterqueue.BUILD",
    sha256 = "67a761278457ab1f113086449c1938e501f272be7f0fd50be28887c1274fe580",
    strip_prefix = "readerwriterqueue-1.0.1",
    urls = [
        "https://github.com/cameron314/readerwriterqueue/archive/v1.0.1.tar.gz",
    ],
)
