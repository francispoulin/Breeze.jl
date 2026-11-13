# Resolve an isolated Julia 1.12 environment without changing the tracked benchmark project.
using Pkg, TOML

repository = realpath(joinpath(@__DIR__, "..", "..", ".."))
environment = ENV["BREEZE_FIR_ENV"]
project = TOML.parsefile(joinpath(repository, "benchmarking", "distributed", "Project.toml"))
project["sources"]["Breeze"]["path"] = repository
open(joinpath(environment, "Project.toml"), "w") do io
    TOML.print(io, project; sorted=true)
end

if !isfile(joinpath(environment, "Manifest.toml"))
    manifest = TOML.parsefile(joinpath(repository, "benchmarking", "distributed", "Manifest.toml"))
    manifest["deps"]["Breeze"][1]["path"] = repository
    open(joinpath(environment, "Manifest.toml"), "w") do io
        TOML.print(io, manifest; sorted=true)
    end
end

# Both MPI.jl and transitive OpenMPI_jll users must load Fir's CUDA-aware system MPI.
mpi_library = joinpath(ENV["EBROOTOPENMPI"], "lib", "libmpi.so")
preferences = Dict(
    "MPIPreferences" => Dict("_format" => "1.0", "abi" => "OpenMPI", "binary" => "system",
                             "libmpi" => mpi_library, "mpiexec" => "mpiexec",
                             "preloads" => String[], "cclibs" => String[]),
    "OpenMPI_jll" => Dict("libmpi_path" => mpi_library))
open(joinpath(environment, "LocalPreferences.toml"), "w") do io
    TOML.print(io, preferences; sorted=true)
end

Pkg.activate(environment)
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
