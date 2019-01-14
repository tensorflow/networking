# Search paths for the MPI header and library files.

# TODO(jbedorf): Make this folder configurable
MPI_ROOT_FOLDER="/usr/local/"

def mpi_copts():
    return ["-I"+MPI_ROOT_FOLDER+"/include"]

def mpi_linkopts():
    return ["-L"+MPI_ROOT_FOLDER+"/lib -lmpi"]
