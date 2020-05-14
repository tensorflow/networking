#FROM tensorflow/tensorflow:nightly-custom-op-ubuntu16
FROM byronyi/tensorflow:ubuntu16.04-manylinux2014

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        libibverbs-dev \
        librdmacm-dev \
        && \
    rm -rf '/var/lib/apt/lists/*'

# Install bazel
ARG BAZEL_VERSION=1.2.1
ARG BAZEL_INSTALLER="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
RUN curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_INSTALLER}" && \
    chmod +x ${BAZEL_INSTALLER} && \
    ./${BAZEL_INSTALLER} && \
    rm ${BAZEL_INSTALLER}

ADD . /tf_networking
RUN cd /tf_networking && \
    python3 third_party/tensorflow/configure.py && \
    bazel build \
    -c opt \
    --cxxopt=-std=gnu++14 \
    --crosstool_top=@//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010:toolchain \
    //tensorflow_networking:libtensorflow_networking.so && \
    cp bazel-bin/tensorflow_networking/libtensorflow_networking.so tensorflow_networking && \
    python3.6 setup.py bdist_wheel && \
    pip3.6 install dist/*.whl
