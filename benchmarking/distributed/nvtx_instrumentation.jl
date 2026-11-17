# Benchmark-only instrumentation: re-evaluate selected host methods from the checkout's
# (or Oceananigans') source, adding NVTX ranges to their bodies. No scientific expression is replaced, no
# device synchronization is added, and kernels and package dependencies are untouched.
# Method replacement is intentional and confined to this profiling process. Enable before
# constructing/warming up the model, never in a production or Reactant tracing process.
module BreezeNVTX

using Breeze
using Oceananigans
using NVTX

const enabled = Ref(false)
const domains = Dict(:Breeze => NVTX.Domain("Breeze"), :Oceananigans => NVTX.Domain("Oceananigans"))

function function_name(signature)
    signature isa Symbol && return signature
    signature isa GlobalRef && return signature.name
    signature isa Expr || return nothing
    signature.head in (:call, :where, :(::)) && return function_name(first(signature.args))
    signature.head === :. && return last(signature.args).value
    return nothing
end

function range_expression(message, expression, domain)
    return Expr(:macrocall, GlobalRef(NVTX, Symbol("@range")), LineNumberNode(0),
                message, Expr(:(=), :domain, domain), expression)
end

# These ranges measure host submission or blocking, not synchronous GPU execution.
# CUDA correlation IDs are needed to attribute the asynchronously launched GPU work.
function annotate_calls(expression, domain)
    expression isa Expr || return expression
    expression.head in (:quote, :macrocall) && return expression
    rewritten = Expr(expression.head, map(argument -> annotate_calls(argument, domain), expression.args)...)
    if expression.head === :call
        name = function_name(first(expression.args))
        if name in (:fill_halo_regions!, :solve!, :solve_for_anelastic_pressure!, :solve_for_pressure!)
            return range_expression("call/$(name)", rewritten, domain)
        end
    end
    return rewritten
end

function annotate_definitions!(target_module, expression, names, found, domain)
    expression isa Expr || return
    if expression.head === :function
        name = function_name(first(expression.args))
        name in names || return
        definition = deepcopy(expression)
        definition.args[2] = range_expression("$(nameof(target_module))/$(name)",
                                              annotate_calls(definition.args[2], domain), domain)
        Core.eval(target_module, definition)
        push!(found, name)
    elseif expression.head in (:toplevel, :block)
        for child in expression.args
            annotate_definitions!(target_module, child, names, found, domain)
        end
    elseif expression.head === :macrocall &&
           first(expression.args) in (GlobalRef(Core, Symbol("@doc")), Symbol("@doc"))
        # Parse docstrings, but never descend into @kernel or other macro definitions.
        annotate_definitions!(target_module, last(expression.args), names, found, domain)
    end
    return nothing
end

function breeze_sources()
    return (
        (Breeze.AtmosphereModels, "AtmosphereModels/update_atmosphere_model_state.jl",
         (:update_state!, :compute_velocities!, :compute_momentum_tendencies!,
          :compute_auxiliary_variables!, :compute_auxiliary_thermodynamic_variables!,
          :compute_tendencies!)),
        (Breeze.CompressibleEquations, "CompressibleEquations/compressible_time_stepping.jl",
         (:compute_total_density!, :compute_auxiliary_dynamics_variables!)),
        (Breeze.CompressibleEquations, "CompressibleEquations/compressible_density_tendency.jl",
         (:compute_dynamics_tendency!,)),
        (Breeze.PotentialTemperatureFormulations, "PotentialTemperatureFormulations/potential_temperature_tendency.jl",
         (:compute_thermodynamic_tendency!,)),
        (Breeze.TimeSteppers, "TimeSteppers/ssp_runge_kutta_3.jl",
         (:time_step!, :store_initial_state!, :ssp_rk3_substep!)),
        (Breeze.TimeSteppers, "TimeSteppers/acoustic_runge_kutta_3.jl",
         (:time_step!, :acoustic_rk3_substep!, :scalar_rk3_substep!)),
        (Breeze.CompressibleEquations, "CompressibleEquations/acoustic_substepping.jl",
         (:freeze_linearization_state!, :refresh_linearization_basic_state!,
          :initialize_stage_perturbations!, :acoustic_substep!, :acoustic_rk3_substep_loop!,
          :apply_divergence_damping!, :finalize_time_averaged_velocity!)),
        (Breeze.AnelasticEquations, "AnelasticEquations/anelastic_time_stepping.jl",
         (:compute_pressure_correction!, :make_pressure_correction!)),
        (Breeze.AnelasticEquations, "AnelasticEquations/anelastic_pressure_solver.jl",
         (:solve_for_anelastic_pressure!, :compute_anelastic_source_term!)),
    )
end

# Oceananigans' NonhydrostaticModel with the RungeKutta3 time stepper. Breeze has its own
# time steppers, so these methods are never called by Breeze models.
function oceananigans_sources()
    NonhydrostaticModels = Oceananigans.Models.NonhydrostaticModels
    return (
        (Oceananigans.TimeSteppers, "TimeSteppers/runge_kutta_3.jl", (:time_step!,)),
        (NonhydrostaticModels, "Models/NonhydrostaticModels/nonhydrostatic_rk3_substep.jl",
         (:pressure_correction_rk3_substep!,)),
        (NonhydrostaticModels, "Models/NonhydrostaticModels/update_nonhydrostatic_model_state.jl",
         (:update_state!, :compute_auxiliaries!)),
        (NonhydrostaticModels, "Models/NonhydrostaticModels/compute_nonhydrostatic_tendencies.jl",
         (:compute_tendencies!, :compute_interior_tendency_contributions!)),
        (NonhydrostaticModels, "Models/NonhydrostaticModels/pressure_correction.jl",
         (:compute_pressure_correction!, :make_pressure_correction!)),
        (NonhydrostaticModels, "Models/NonhydrostaticModels/solve_for_pressure.jl",
         (:solve_for_pressure!, :compute_source_term!)),
    )
end

"""
    enable!(package=:Breeze)

Annotate the host methods of `package`, either `:Breeze` (AtmosphereModel) or
`:Oceananigans` (NonhydrostaticModel), with NVTX ranges in a domain of the same name.
"""
function enable!(package=:Breeze)
    enabled[] && error("NVTX instrumentation has already been enabled")
    package in (:Breeze, :Oceananigans) || error("Unknown NVTX instrumentation package: $package")
    root = package === :Breeze ? pkgdir(Breeze) : pkgdir(Oceananigans)
    sources = package === :Breeze ? breeze_sources() : oceananigans_sources()
    domain = domains[package]
    for (target_module, relative_path, names) in sources
        path = joinpath(root, "src", relative_path)
        found = Set{Symbol}()
        annotate_definitions!(target_module, Meta.parseall(read(path, String); filename=path), names, found, domain)
        missing = setdiff(Set(names), found)
        isempty(missing) || error("NVTX target methods missing from $path: $missing")
    end
    enabled[] = true
    @info "NVTX host instrumentation enabled" package NVTX.isactive()
    return nothing
end

end # module
