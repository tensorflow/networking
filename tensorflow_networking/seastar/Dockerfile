#FROM tensorflow/tensorflow:nightly-custom-op-ubuntu16
FROM byronyi/tensorflow:ubuntu16.04-manylinux2014

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        libibverbs-dev \
        librdmacm-dev \
        && \
    rm -rf '/var/lib/apt/lists/*'

# Install bazel
ARG BAZEL_VERSION=3.1.0
ARG BAZEL_INSTALLER="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
RUN curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_INSTALLER}" && \
    chmod +x ${BAZEL_INSTALLER} && \
    ./${BAZEL_INSTALLER} && \
    rm ${BAZEL_INSTALLER}

RUN apt-get update
RUN apt-get install git -y
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3
RUN ln -sf /usr/local/bin/pip3.6 /usr/local/bin/pip3

RUN git clone https://github.com/tensorflow/networking.git

RUN cd /networking && \
    python3 third_party/tensorflow/configure.py && \
    bazel build \
    -c opt \
    --cxxopt=-std=gnu++14 \
    --crosstool_top=@//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2014:toolchain \
    //tensorflow_networking:libtensorflow_networking.so && \
    cp bazel-bin/tensorflow_networking/libtensorflow_networking.so tensorflow_networking && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl
