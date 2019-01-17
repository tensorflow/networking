# TensorFlow Networking


This repository is for platform-specific networking extensions to core TensorFlow and related
utilities (e.g. testing).

The design goal is to work towards separately compilable plugins, but initially we'll just be porting the
networking related contrib directories since TensorFlow 2.0 will be dropping contrib.

## Building

Currently support building GDR and MPI extensions:

#### GDR

Using Bazel:

```bash
bazel build -c opt //tensorflow_networking/gdr:gdr_server_lib
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/gdr/Dockerfile .
```

####  MPI

Note: This is for the `grpc+mpi` extension

For the MPI extension the location to the MPI library has to be configured. For this run the  `configure` script. The script will attempt to find the `mpirun` binary and deduce the include and library paths. If `mpirun` is not installed in your PATH or you want to use another location you can specify this via the `MPI_HOME` variable. The configure script will create symlinks to the relevant MPI header and library files inside the `third_party/mpi` folder.

Using Bazel:

By manually answering the relevant configuration questions
```bash
./configure
bazel build -c opt //tensorflow_networking/mpi:mpi_server_lib
```
or by preset answers to the configuration questions
```bash
MPI_HOME=<path to mpi folder root> TF_NEED_MPI=1 ./configure
bazel build -c opt //tensorflow_networking/mpi:mpi_server_lib
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/mpi/Dockerfile .
```
