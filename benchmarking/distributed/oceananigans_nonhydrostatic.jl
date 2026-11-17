#####
##### Oceananigans NonhydrostaticModel reference workload for the distributed benchmark.
#####
##### This is the `rectilinear_buoyancy` NonhydrostaticModel case of the Oceananigans benchmark
##### suite (Oceananigans.jl `benchmarking/src/rectilinear_buoyancy.jl` with the settings from
##### `benchmarking/src/suite/plan.py`): 1 km × 1 km × 10 m cells, periodic in x and y, WENO5
##### momentum and tracer advection, one buoyancy tracer and RungeKutta3. On several ranks the
##### pressure solve is Oceananigans' DistributedFFTBasedPoissonSolver, the same transposes as
##### Breeze's anelastic solver. It lets the distributed driver time and profile Breeze and
##### Oceananigans with the same Oceananigans version, communicator and Nsight capture window.
#####
##### The suite uses Float64 and Δt = 10 s.
#####

function oceananigans_nonhydrostatic(arch; float_type = Float64, Nx = 512, Ny = 512, Nz = 512)
    Lx, Ly, Lz = 1000Nx, 1000Ny, 10Nz
    grid = RectilinearGrid(arch, float_type;
                           size = (Nx, Ny, Nz), halo = (3, 3, 3),
                           topology = (Periodic, Periodic, Bounded),
                           x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

    advection = WENO(float_type; order=5)
    model = NonhydrostaticModel(grid; momentum_advection = advection, tracer_advection = advection,
                                buoyancy = BuoyancyTracer(), tracers = :b, closure = nothing,
                                coriolis = nothing, timestepper = :RungeKutta3)

    # Stable stratification plus a smooth perturbation that repeats across x slabs.
    bᵢ(x, y, z) = 1e-5 * z + 1e-6 * cospi(x / 64000) * cospi(y / 64000) * sinpi(z / 5120)
    set!(model; b=bᵢ, u=1//10, v=1//20)

    return model
end
