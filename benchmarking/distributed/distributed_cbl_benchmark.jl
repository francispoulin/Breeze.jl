#####
##### Multi-GPU (MPI) time-stepping benchmark for the convective boundary layer (CBL) case.
#####
##### Reuses `convective_boundary_layer` from `benchmarking/src`, but builds the model on an
##### Oceananigans `Distributed` architecture (one GPU per MPI rank) and times plain `time_step!`
##### calls, with no Reactant or Simulation overhead. See README.md in this folder.
#####
##### Usage, from the root of the repository:
#####
#####   mpiexec -n 4 julia --project=benchmarking/distributed \
#####       benchmarking/distributed/distributed_cbl_benchmark.jl \
#####       --dynamics=compressible_splitexplicit --size=512x512x128 --partition=2x2
#####
##### Options (all `--name=value`):
#####   --dynamics         compressible_explicit | compressible_splitexplicit   (default compressible_splitexplicit)
#####   --size             global grid, NxNyxNz                                (default 256x256x128)
#####   --partition        ranks in x and y, RxxRy; must multiply to #ranks    (default <#ranks>x1)
#####   --time_steps       timed steps                                         (default 100)
#####   --warmup_steps     untimed steps (includes compilation)                (default 10)
#####   --time_step        Δt in seconds                                       (default 0.05)
#####   --substeps         acoustic substeps (split-explicit only)             (default 12)
#####   --float_type       Float32 | Float64                                   (default Float32)
#####   --simplified       true | false: drop geostrophic forcing and drag BCs (default false)
#####   --results_file     append one CSV line of results to this file         (default: none)
#####

using MPI
MPI.Init()

using CUDA
using Oceananigans
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.TimeSteppers: time_step!
using Breeze
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization, ExplicitTimeStepping
using Printf

# Include the case directly rather than loading the whole BreezeBenchmarks module, which pulls in
# Reactant, Enzyme, AMDGPU and Metal.
include(joinpath(@__DIR__, "..", "src", "convective_boundary_layer.jl"))

const comm   = MPI.COMM_WORLD
const rank   = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

function option(name, default)
    for argument in ARGS
        startswith(argument, "--$name=") && return String(split(argument, "="; limit=2)[2])
    end
    return default
end

parse_dimensions(text) = Tuple(parse.(Int, split(text, ['x', '×', '*'])))

dynamics_name = option("dynamics", "compressible_splitexplicit")
Nx, Ny, Nz    = parse_dimensions(option("size", "256x256x128"))
Rx, Ry        = parse_dimensions(option("partition", "$(nranks)x1"))
time_steps    = parse(Int, option("time_steps", "100"))
warmup_steps  = parse(Int, option("warmup_steps", "10"))
Δt            = parse(Float64, option("time_step", "0.05"))
substeps      = parse(Int, option("substeps", "12"))
FT            = option("float_type", "Float32") == "Float64" ? Float64 : Float32
simplified    = option("simplified", "false") == "true"
results_file  = option("results_file", "")

Rx * Ry == nranks || error("--partition=$(Rx)x$(Ry) needs $(Rx * Ry) ranks but mpiexec started $nranks")

dynamics = if dynamics_name == "compressible_explicit"
    CompressibleDynamics(ExplicitTimeStepping())
elseif dynamics_name == "compressible_splitexplicit"
    CompressibleDynamics(SplitExplicitTimeDiscretization(FT; substeps))
elseif dynamics_name == "anelastic"
    # AnelasticDynamics builds a serial FourierTridiagonalPoissonSolver on each rank's local grid,
    # which does not solve the global pressure problem. Refuse rather than time a wrong solve.
    error("anelastic dynamics is not distributed in this Breeze version; see README.md (distributed pressure solver, Breeze PR 1157)")
else
    error("Unknown --dynamics=$dynamics_name")
end

arch = Distributed(GPU(); partition = Partition(Rx, Ry))

model = convective_boundary_layer(arch; float_type = FT, Nx, Ny, Nz, dynamics, simplified)

rank == 0 && @info "Distributed CBL benchmark" dynamics_name FT (Nx, Ny, Nz) partition=(Rx, Ry) nranks Δt time_steps

for _ in 1:warmup_steps
    time_step!(model, FT(Δt))
end

CUDA.synchronize()
MPI.Barrier(comm)
start_time = time_ns()

for _ in 1:time_steps
    time_step!(model, FT(Δt))
end

CUDA.synchronize()
MPI.Barrier(comm)
elapsed = MPI.Allreduce((time_ns() - start_time) / 1e9, max, comm)

# A blown-up run gives meaningless throughput, so check the solution is finite and report max |w|.
local_max_w = maximum(abs, interior(model.velocities.w))
max_w = MPI.Allreduce(Float64(local_max_w), max, comm)

if rank == 0
    total_points      = Nx * Ny * Nz
    seconds_per_step  = elapsed / time_steps
    points_per_second = total_points / seconds_per_step

    # @printf needs a single string literal as its format, so this stays on one line.
    @printf("RESULT dynamics=%s float=%s size=%dx%dx%d partition=%dx%d gpus=%d steps=%d seconds_per_step=%.4e points_per_second=%.4e points_per_second_per_gpu=%.4e max_abs_w=%.3e\n",
            dynamics_name, FT, Nx, Ny, Nz, Rx, Ry, nranks, time_steps,
            seconds_per_step, points_per_second, points_per_second / nranks, max_w)

    isfinite(max_w) || @warn "max |w| is not finite: the run blew up, so the timing above is not meaningful"

    if !isempty(results_file)
        write_header = !isfile(results_file)
        open(results_file, "a") do io
            write_header && println(io, "dynamics,float,Nx,Ny,Nz,Rx,Ry,gpus,steps,seconds_per_step,points_per_second,max_abs_w")
            @printf(io, "%s,%s,%d,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e\n",
                    dynamics_name, FT, Nx, Ny, Nz, Rx, Ry, nranks, time_steps,
                    seconds_per_step, points_per_second, max_w)
        end
    end
end

MPI.Barrier(comm)
