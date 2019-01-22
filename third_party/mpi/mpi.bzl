# OpenMPI and Mvapich/mpich require different headers
# based on the configuration options return one or the other

def mpi_hdr():
    return if_openmpi(
        ["mpi.h", "mpi_portable_platform.h"],
        ["mpi.h", "mpio.h", "mpicxx.h"],
    )

def if_openmpi(if_true, if_false = []):
    return select({
        "//tensorflow_networking:mpi_library_is_openmpi_based": if_true,
        "//conditions:default": if_false,
    })
