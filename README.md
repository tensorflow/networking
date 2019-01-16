TensorFlow Networking
===

This repository is for platform-specific networking extensions to core TensorFlow and related
utilities (e.g. testing).

The design goal is to work towards separately compilable plugins, but initially we'll just be porting the
networking related contrib directories since TensorFlow 2.0 will be dropping contrib.

Building
===

Currently support building GDR extension:

Using Bazel:

```bash
bazel build -c opt //tensorflow_networking/gdr:gdr_server_lib
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/gdr/Dockerfile .
```
