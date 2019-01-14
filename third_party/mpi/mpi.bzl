#OpenMPI and Mvapich/mpich require different headers
#based on the configuration options return one or the other

# TODO(jbedorf): Make this folder configurable
MPI_ROOT_FOLDER="/usr/local/"

def mpi_copts():
    return if_mpi(["-I"+MPI_ROOT_FOLDER+"/include"])

def mpi_linkopts():
    return if_mpi(["-L"+MPI_ROOT_FOLDER+"/lib -lmpi"])

def mpi_defines():
    return if_mpi(["TENSORFLOW_USE_MPI"])

def if_mpi(if_true, if_false = []):
    return select({
        "//tensorflow_networking:with_mpi_support2": if_true,
        "//conditions:default": if_false,
})

