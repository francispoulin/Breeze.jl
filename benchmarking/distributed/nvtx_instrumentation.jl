# Benchmark-only instrumentation: re-evaluate selected host methods from the checkout's
# source, adding NVTX ranges to their bodies. No scientific expression is replaced, no
# device synchronization is added, and kernels and package dependencies are untouched.
# Method replacement is intentional and confined to this profiling process. Enable before
# constructing/warming up the model, never in a production or Reactant tracing process.
module BreezeNVTX

using Breeze
using NVTX

const enabled = Ref(false)
const domain = NVTX.Domain("Breeze")

function function_name(signature)
    signature isa Symbol && return signature
    signature isa GlobalRef && return signature.name
    signature isa Expr || return nothing
    signature.head in (:call, :where, :(::)) && return function_name(first(signature.args))
    signature.head === :. && return last(signature.args).value
    return nothing
end

function range_expression(message, expression)
    return Expr(:macrocall, GlobalRef(NVTX, Symbol("@range")), LineNumberNode(0),
                message, Expr(:(=), :domain, domain), expression)
end

# These ranges measure host submission or blocking, not synchronous GPU execution.
# CUDA correlation IDs are needed to attribute the asynchronously launched GPU work.
function annotate_calls(expression)
    expression isa Expr || return expression
    expression.head in (:quote, :macrocall) && return expression
    rewritten = Expr(expression.head, map(annotate_calls, expression.args)...)
    if expression.head === :call
        name = function_name(first(expression.args))
        if name in (:fill_halo_regions!, :solve!, :solve_for_anelastic_pressure!)
            return range_expression("call/$(name)", rewritten)
        end
    end
    return rewritten
end

function annotate_definitions!(target_module, expression, names, found)
    expression isa Expr || return
    if expression.head === :function
        name = function_name(first(expression.args))
        name in names || return
        definition = deepcopy(expression)
        definition.args[2] = range_expression("$(nameof(target_module))/$(name)",
                                              annotate_calls(definition.args[2]))
        Core.eval(target_module, definition)
        push!(found, name)
    elseif expression.head in (:toplevel, :block)
        for child in expression.args
            annotate_definitions!(target_module, child, names, found)
        end
    elseif expression.head === :macrocall &&
           first(expression.args) in (GlobalRef(Core, Symbol("@doc")), Symbol("@doc"))
        # Parse docstrings, but never descend into @kernel or other macro definitions.
        annotate_definitions!(target_module, last(expression.args), names, found)
    end
    return nothing
end

function enable!()
    enabled[] && error("Breeze NVTX instrumentation has already been enabled")
    sources = (
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
    for (target_module, relative_path, names) in sources
        path = joinpath(pkgdir(Breeze), "src", relative_path)
        found = Set{Symbol}()
        annotate_definitions!(target_module, Meta.parseall(read(path, String); filename=path), names, found)
        missing = setdiff(Set(names), found)
        isempty(missing) || error("NVTX target methods missing from $path: $missing")
    end
    enabled[] = true
    @info "Breeze NVTX host instrumentation enabled" NVTX.isactive()
    return nothing
end

end # module
