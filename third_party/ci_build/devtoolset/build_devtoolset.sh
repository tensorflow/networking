#!/bin/bash -eu
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Builds a cross-compiler targeting manylinux 2014 (glibc 2.17 / libstdc++ 4.8).

VERSION="$1"
TARGET="$2"

case "${VERSION}" in
devtoolset-7)
  LIBSTDCXX_VERSION="6.0.24"
  ;;
devtoolset-8)
  LIBSTDCXX_VERSION="6.0.25"
  ;;
*)
  echo "Usage: $0 {devtoolset-7|devtoolset-8} <target-directory>"
  exit 1
  ;;
esac

mkdir -p "${TARGET}"
# Download binary glibc 2.17 release.
curl -O "http://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6_2.17-0ubuntu5.1_amd64.deb" && \
    unar "libc6_2.17-0ubuntu5.1_amd64.deb" && \
    tar -C "${TARGET}" -xf "libc6_2.17-0ubuntu5.1_amd64/data.tar.gz" && \
    rm -rf "libc6_2.17-0ubuntu5.1_amd64.deb" "libc6_2.17-0ubuntu5.1_amd64"
curl -O "http://old-releases.ubuntu.com/ubuntu/pool/main/e/eglibc/libc6-dev_2.17-0ubuntu5.1_amd64.deb" && \
    unar "libc6-dev_2.17-0ubuntu5.1_amd64.deb" && \
    tar -C "${TARGET}" -xf "libc6-dev_2.17-0ubuntu5.1_amd64/data.tar.gz" && \
    rm -rf "libc6-dev_2.17-0ubuntu5.1_amd64.deb" "libc6-dev_2.17-0ubuntu5.1_amd64"

# Put the current kernel headers from ubuntu in place.
ln -s "/usr/include/linux" "${TARGET}/usr/include/linux"
ln -s "/usr/include/asm-generic" "${TARGET}/usr/include/asm-generic"
ln -s "/usr/include/x86_64-linux-gnu/asm" "${TARGET}/usr/include/asm"

# Symlinks in the binary distribution are set up for installation in /usr, we
# need to fix up all the links to stay within ${TARGET}.
/fixlinks.sh "${TARGET}"

# Patch to allow non-glibc 2.17 compatible builds to work.
sed -i '54i#define TCP_USER_TIMEOUT 18' "${TARGET}/usr/include/netinet/tcp.h"

# Download binary libstdc++ 4.8 release we are going to link against.
# We only need the shared library, as we're going to develop against the
# libstdc++ provided by devtoolset.
curl -O "http://old-releases.ubuntu.com/ubuntu/pool/main/g/gcc-4.8/libstdc++6_4.8.1-10ubuntu9_amd64.deb" && \
    unar "libstdc++6_4.8.1-10ubuntu9_amd64.deb" && \
    tar -C "${TARGET}" -xf "libstdc++6_4.8.1-10ubuntu9_amd64/data.tar.gz" "./usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.18" && \
    rm -rf "libstdc++6_4.8.1-10ubuntu9_amd64.deb" "libstdc++6_4.8.1-10ubuntu9_amd64"

mkdir -p "${TARGET}-src"
cd "${TARGET}-src"

# Build a devtoolset cross-compiler based on our glibc 2.17 sysroot setup.
case "${VERSION}" in
devtoolset-7)
  curl -O "http://vault.centos.org/centos/7/sclo/Source/rh/devtoolset-7/devtoolset-7-gcc-7.3.1-5.16.el7.src.rpm"
  rpm2cpio "devtoolset-7-gcc-7.3.1-5.16.el7.src.rpm" | cpio -idm
  tar -xf "gcc-7.3.1-20180303.tar.bz2" --strip 1
  ;;
devtoolset-8)
  curl -O "http://vault.centos.org/centos/7/sclo/Source/rh/devtoolset-8/devtoolset-8-gcc-8.3.1-3.el7.src.rpm"
  rpm2cpio "devtoolset-8-gcc-8.3.1-3.el7.src.rpm" | cpio -idm
  tar -xf "gcc-8.3.1-20190311.tar.xz" --strip 1
  ;;
esac

# Apply the devtoolset patches to gcc.
/rpm-patch.sh "gcc.spec"

sed -i 's/ftp:\/\/gcc.gnu.org/https:\/\/mirror.math.princeton.edu/g' ./contrib/download_prerequisites
./contrib/download_prerequisites

mkdir -p "${TARGET}-build"
cd "${TARGET}-build"

"${TARGET}-src/configure" \
      --prefix="${TARGET}/usr" \
      --with-sysroot="${TARGET}" \
      --disable-bootstrap \
      --disable-libmpx \
      --disable-libsanitizer \
      --disable-libunwind-exceptions \
      --disable-libunwind-exceptions \
      --disable-lto \
      --disable-multilib \
      --enable-__cxa_atexit \
      --enable-gnu-indirect-function \
      --enable-gnu-unique-object \
      --enable-initfini-array \
      --enable-languages="c,c++" \
      --enable-linker-build-id \
      --enable-plugin \
      --enable-shared \
      --enable-threads=posix \
      --with-default-libstdcxx-abi="gcc4-compatible" \
      --with-gcc-major-version-only \
      --with-linker-hash-style="gnu" \
      --with-tune="generic" \
      && \
    make -j && \
    make install

# Create the devtoolset libstdc++ linkerscript that links dynamically against
# the system libstdc++ 4.8 and provides all other symbols statically.
mv "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}" \
   "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}.backup"
echo -e "OUTPUT_FORMAT(elf64-x86-64)\nINPUT ( libstdc++.so.6.0.18 -lstdc++_nonshared44 )" \
   > "${TARGET}/usr/lib64/libstdc++.so.${LIBSTDCXX_VERSION}"
cp "./x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++_nonshared44.a" \
   "${TARGET}/usr/lib"

# Clean up
rm -rf "${TARGET}-build"
rm -rf "${TARGET}-src"
