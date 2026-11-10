# Run with the distributed benchmark project. Compare full GPU fields before and
# after enabling host annotations, reusing the same model and FFT plan. Separate
# cuFFT plans need not produce bitwise identical small anelastic velocities.
using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TimeSteppers: update_state!
using CUDA
using NVTX
using Test

include(joinpath(@__DIR__, "..", "src", "convective_boundary_layer.jl"))
include(joinpath(@__DIR__, "nvtx_instrumentation.jl"))

function final_fields(model, initial_fields, initial_clock)
    for (field, initial) in zip(prognostic_fields(model), initial_fields)
        copyto!(parent(field), initial)
    end
    for name in fieldnames(typeof(model.clock))
        setproperty!(model.clock, name, getproperty(initial_clock, name))
    end
    update_state!(model)
    for _ in 1:3
        time_step!(model, 0.005f0)
    end
    CUDA.synchronize()
    return map(field -> Array(interior(field)),
               merge(prognostic_fields(model), model.velocities))
end

dynamics = (nothing,
            CompressibleDynamics(ExplicitTimeStepping()),
            CompressibleDynamics(SplitExplicitTimeDiscretization(Float32; substeps=12)))
models = map(configuration -> convective_boundary_layer(GPU(); Nx = 16, Ny = 16, Nz = 16,
              dynamics = configuration, random_perturbation = false), dynamics)
initial_fields = map(model -> map(field -> copy(parent(field)), prognostic_fields(model)), models)
initial_clocks = map(model -> deepcopy(model.clock), models)
baseline = map(final_fields, models, initial_fields, initial_clocks)
repeated = map(final_fields, models, initial_fields, initial_clocks)

@testset "Uninstrumented repeat control" begin
    for (expected, actual) in zip(baseline, repeated)
        for name in keys(expected)
            identical = actual[name] == expected[name]
            @info "Repeat control" name identical maximum_absolute_difference=maximum(abs, actual[name] - expected[name])
            @test identical
        end
    end
end

NVTX.activate()
BreezeNVTX.enable!()

# Call through invokelatest because enabling instrumentation replaces methods.
@testset "NVTX preserves GPU fields" begin
    for (model, initial, clock, expected) in zip(models, initial_fields, initial_clocks, baseline)
        actual = Base.invokelatest(final_fields, model, initial, clock)
        @test keys(actual) == keys(expected)
        for name in keys(expected)
            @test all(isfinite, actual[name])
            identical = actual[name] == expected[name]
            @info "Annotation comparison" name identical maximum_absolute_difference=maximum(abs, actual[name] - expected[name])
            @test identical
        end
    end
end
println("BREEZE_NVTX_VALIDATION_OK")
