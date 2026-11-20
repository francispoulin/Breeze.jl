#####
##### Oceananigans reference workloads for the distributed benchmark.
#####
##### These are the `rectilinear_buoyancy` cases of the Oceananigans benchmark suite
##### (Oceananigans.jl `benchmarking/src/rectilinear_buoyancy.jl`, with the settings from
##### `benchmarking/src/suite/plan.py`): 1 km × 1 km × 10 m cells, periodic in x and y, WENO5
##### momentum and tracer advection and one buoyancy tracer.
#####
##### - `NonhydrostaticModel` with RungeKutta3. On several ranks the pressure solve is
#####   Oceananigans' DistributedFFTBasedPoissonSolver, the same transposes as Breeze's
#####   anelastic solver.
##### - `HydrostaticFreeSurfaceModel` with SplitRungeKutta3 and a SplitExplicitFreeSurface
#####   with 30 substeps; on GPUs the suite extends the free-surface halos.
#####
##### They let the distributed driver time and profile Breeze and Oceananigans with the same
##### Oceananigans version, communicator and Nsight capture window. The suite uses Float64 and
##### Δt = 10 s.
#####

function oceananigans_rectilinear_buoyancy(arch, model_type; float_type = Float64,
                                           Nx = 512, Ny = 512, Nz = 512,
                                           free_surface_substeps = 30, extend_free_surface_halos = true)
    Lx, Ly, Lz = 1000Nx, 1000Ny, 10Nz
    grid = RectilinearGrid(arch, float_type;
                           size = (Nx, Ny, Nz), halo = (3, 3, 3),
                           topology = (Periodic, Periodic, Bounded),
                           x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

    advection = WENO(float_type; order=5)
    common = (; momentum_advection = advection, tracer_advection = advection,
                buoyancy = BuoyancyTracer(), tracers = :b, closure = nothing, coriolis = nothing)

    model = if model_type === :hydrostatic
        free_surface = SplitExplicitFreeSurface(; substeps = free_surface_substeps,
                                                  extend_halos = extend_free_surface_halos)
        HydrostaticFreeSurfaceModel(grid; common..., free_surface, timestepper = :SplitRungeKutta3)
    elseif model_type === :nonhydrostatic
        NonhydrostaticModel(grid; common..., timestepper = :RungeKutta3)
    else
        error("Unknown Oceananigans model type: $model_type")
    end

    # Stable stratification plus a smooth perturbation that repeats across x slabs.
    bᵢ(x, y, z) = 1e-5 * z + 1e-6 * cospi(x / 64000) * cospi(y / 64000) * sinpi(z / 5120)
    set!(model; b=bᵢ, u=1//10, v=1//20)

    return model
end
