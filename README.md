# TensorFlow Networking


This repository is for platform-specific networking extensions to core TensorFlow and related
utilities (e.g. testing).

The design goal is to work towards separately compilable plugins, but initially we'll just be porting the
networking related contrib directories since TensorFlow 2.0 will be dropping contrib.

## Building

Currently support building GDR, VERBS, and MPI extensions:

#### GDR

Using Bazel:

```bash
bazel build -c opt //tensorflow_networking/gdr:gdr_server_lib
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/gdr/Dockerfile .
```

#### VERBS

Using Bazel:

```bash
bazel build -c opt //tensorflow_networking/verbs:verbs_server_lib
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/verbs/Dockerfile .
```

####  MPI


For the MPI extensions the location to the MPI library has to be configured. The `configure` script is used to setup this configuration. The script will attempt to find the location of the `mpirun` binary and from there deduce the include and library paths. You can use the `MPI_HOME` environment variable if `mpirun` is not installed in your PATH or you want to use another base path for the MPI library. The configure script will create symbolic links inside the `third_party/mpi` folder to the relevant MPI header and library files. Furthermore the script will determine if your MPI installation is based on `OpenMPI` or on `MPICH` and sets this in the `.tf_networking_configure.bazelrc` file.

#####  `grpc+mpi` extension

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


#####  `MPI collectives` extension

Using Bazel:

By manually answering the relevant configuration questions
```bash
./configure
bazel build -c opt //tensorflow_networking/mpi_collectives:all
```

Using Docker:

```bash
docker build -t tf_networking -f tensorflow_networking/mpi_collectives/Dockerfile .
```

#####  `grpc+seastar` extension

Using Bazel:

```bash
bazel build -c opt --copt='-std=gnu++14' //tensorflow_networking:libtensorflow_networking.so
