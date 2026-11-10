#####
##### Multi-GPU (MPI or NCCL) time-stepping benchmark for the convective boundary layer (CBL) case.
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
#####   --dynamics         anelastic (1 GPU only) | compressible_explicit | compressible_splitexplicit
#####                                                                          (default compressible_splitexplicit)
#####   --size             global grid, NxNyxNz                                (default 256x256x128)
#####   --partition        ranks in x and y, RxxRy; must multiply to #ranks    (default <#ranks>x1)
#####   --comm             MPI | NCCL; one rank uses the serial GPU baseline (default MPI)
#####   --time_steps       timed steps                                         (default 100)
#####   --warmup_steps     untimed steps (includes compilation)                (default 10)
#####   --time_step        Δt in seconds                                       (default 0.05)
#####   --substeps         acoustic substeps (split-explicit only)             (default 12)
#####   --float_type       Float32 | Float64                                   (default Float32)
#####   --simplified       true | false: drop geostrophic forcing and drag BCs (default false)
#####   --profile          true | false: capture the timed steps with Nsight  (default false)
#####   --results_file     append one CSV line of results to this file         (default: none)
#####

using MPI
MPI.Init()

using CUDA
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.DistributedComputations: Distributed, NCCLDistributed, Partition
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

function advance_time_steps!(model, Δt, time_steps)
    for _ in 1:time_steps
        time_step!(model, Δt)
    end
    CUDA.synchronize()
    return nothing
end

dynamics_name = option("dynamics", "compressible_splitexplicit")
comm_name     = uppercase(option("comm", "MPI"))
Nx, Ny, Nz    = parse_dimensions(option("size", "256x256x128"))
Rx, Ry        = parse_dimensions(option("partition", "$(nranks)x1"))
time_steps    = parse(Int, option("time_steps", "100"))
warmup_steps  = parse(Int, option("warmup_steps", "10"))
Δt            = parse(Float64, option("time_step", "0.05"))
substeps      = parse(Int, option("substeps", "12"))
FT            = option("float_type", "Float32") == "Float64" ? Float64 : Float32
simplified    = option("simplified", "false") == "true"
profile       = parse(Bool, option("profile", "false"))
results_file  = option("results_file", "")

Rx * Ry == nranks || error("--partition=$(Rx)x$(Ry) needs $(Rx * Ry) ranks but mpiexec started $nranks")
comm_name in ("MPI", "NCCL") || error("--comm must be MPI or NCCL")
comm_name == "NCCL" && @eval using NCCL

csv_header = "machine,dynamics,float,Nx,Ny,Nz,Rx,Ry,gpus,steps,seconds_per_step,points_per_second,max_abs_w,communication,fields_finite"
if !isempty(results_file) && isfile(results_file)
    readline(results_file) == csv_header || error("Results CSV header differs; use a fresh --results_file")
end

dynamics = if dynamics_name == "compressible_explicit"
    CompressibleDynamics(ExplicitTimeStepping())
elseif dynamics_name == "compressible_splitexplicit"
    CompressibleDynamics(SplitExplicitTimeDiscretization(FT; substeps))
elseif dynamics_name == "anelastic"
    # AnelasticDynamics builds a serial FourierTridiagonalPoissonSolver on each rank's local grid,
    # which does not solve the global pressure problem on several ranks. Refuse rather than time a
    # wrong solve. On one rank the local grid is the global grid, so it is correct.
    nranks == 1 || error("anelastic dynamics is not distributed in this Breeze version, so it runs on 1 GPU only; see README.md (distributed pressure solver, Breeze PR 1157)")
    nothing  # `convective_boundary_layer` builds AnelasticDynamics when `dynamics` is `nothing`
else
    error("Unknown --dynamics=$dynamics_name")
end

# One rank is the plain serial baseline, with no MPI machinery in the model.
communication_name = nranks == 1 ? "SERIAL" : comm_name
arch = if nranks == 1
    GPU()
elseif comm_name == "NCCL"
    NCCLDistributed(GPU(); partition = Partition(Rx, Ry))
else
    Distributed(GPU(); partition = Partition(Rx, Ry))
end

if nranks > 1 && comm_name == "NCCL"
    nccl_extension = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
    isnothing(nccl_extension) && error("OceananigansNCCLExt did not load")
    arch.communicator isa nccl_extension.NCCLCommunicator || error("NCCL communicator was not selected")
    isnothing(arch.communicator.nccl) && error("NCCL communicator was not initialized")
end
communicator_type = nranks == 1 ? "none" : string(nameof(typeof(arch.communicator)))

model = convective_boundary_layer(arch; float_type = FT, Nx, Ny, Nz, dynamics, simplified)

rank == 0 && @info "Distributed CBL benchmark" dynamics_name FT (Nx, Ny, Nz) partition=(Rx, Ry) nranks Δt time_steps communication_name communicator_type profile

advance_time_steps!(model, FT(Δt), warmup_steps)

rank == 0 && @printf("TIMING_START communication=%s steps=%d\n", communication_name, time_steps)
MPI.Barrier(comm)
start_time = time_ns()

if profile
    @printf("PROFILE_START rank=%d steps=%d\n", rank, time_steps)
    # Warmup compiles this same stepping function. Only the requested timed steps and their
    # final device synchronization are captured; model setup and field checks stay outside.
    CUDA.@profile external=true advance_time_steps!(model, FT(Δt), time_steps)
    @printf("PROFILE_END rank=%d steps=%d\n", rank, time_steps)
else
    advance_time_steps!(model, FT(Δt), time_steps)
end

MPI.Barrier(comm)
elapsed = MPI.Allreduce((time_ns() - start_time) / 1e9, max, comm)
rank == 0 && @printf("TIMING_END communication=%s elapsed_seconds=%.9f\n", communication_name, elapsed)

# A blown-up run gives meaningless throughput, so check the solution is finite and report max |w|.
local_max_w = maximum(abs, interior(model.velocities.w))
max_w = MPI.Allreduce(Float64(local_max_w), max, comm)
local_fields_finite = all(field -> all(isfinite, interior(field)), values(prognostic_fields(model))) &&
                      all(field -> all(isfinite, interior(field)), values(model.velocities))
fields_finite = MPI.Allreduce(Int(local_fields_finite), min, comm) == 1

if rank == 0
    machine           = gethostname()
    total_points      = Nx * Ny * Nz
    seconds_per_step  = elapsed / time_steps
    points_per_second = total_points / seconds_per_step

    # @printf needs a single string literal as its format, so this stays on one line.
    @printf("RESULT machine=%s dynamics=%s float=%s size=%dx%dx%d partition=%dx%d gpus=%d steps=%d seconds_per_step=%.4e points_per_second=%.4e points_per_second_per_gpu=%.4e max_abs_w=%.3e communication=%s fields_finite=%s\n",
            machine, dynamics_name, FT, Nx, Ny, Nz, Rx, Ry, nranks, time_steps,
            seconds_per_step, points_per_second, points_per_second / nranks, max_w, communication_name, string(fields_finite))

    fields_finite || @warn "Non-finite fields: the timing above is not meaningful" max_w

    if !isempty(results_file)
        write_header = !isfile(results_file)
        open(results_file, "a") do io
            write_header && println(io, csv_header)
            @printf(io, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%s,%s\n",
                    machine, dynamics_name, FT, Nx, Ny, Nz, Rx, Ry, nranks, time_steps,
                    seconds_per_step, points_per_second, max_w, communication_name, string(fields_finite))
        end
    end
end

MPI.Barrier(comm)
