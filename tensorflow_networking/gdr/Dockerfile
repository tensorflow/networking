FROM tensorflow/tensorflow:nightly-py3

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        g++ \
        libibverbs-dev \
        librdmacm-dev \
        openjdk-8-jdk \
        unzip \
        zip \
        && \
    rm -rf '/var/lib/apt/lists/*'

# Install bazel
ARG BAZEL_VERSION=1.1.0
ARG BAZEL_INSTALLER="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
RUN curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_INSTALLER}" && \
    chmod +x ${BAZEL_INSTALLER} && \
    ./${BAZEL_INSTALLER} && \
    rm ${BAZEL_INSTALLER}

ADD . /tf_networking
RUN cd /tf_networking && \
    python3 third_party/tensorflow/configure.py && \
    bazel build -c opt //tensorflow_networking:libtensorflow_networking.so && \
    cp bazel-bin/tensorflow_networking/libtensorflow_networking.so tensorflow_networking && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl
