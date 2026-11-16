# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module BilevelJuMPDisjunctiveProgrammingExt

import BilevelJuMP
import DisjunctiveProgramming as DP
import JuMP
import MathOptInterface as MOI

function BilevelJuMP._disjunction_set(disjunct_sets)
    return DP.DisjunctionSet(disjunct_sets)
end

#=
    Reformulating solver

    Wraps the user's solver for the duration of one solve. `copy_to`
    reformulates the `DisjunctionSet` constraints with the mode's
    `gdp_method` in a `GDPModel` and copies the result into the inner
    solver; every other call is forwarded. The returned index map points at
    the inner solver's own indices, and `model.solver` stays the inner
    solver, so results are read from it as for any other mode.
=#

struct _ReformulatingSolver{S<:MOI.ModelLike} <: MOI.AbstractOptimizer
    inner_solver::S
    gdp_method::Any
end

function MOI.copy_to(
    reformulating_solver::_ReformulatingSolver,
    single_level_model::MOI.ModelLike,
)
    gdp_model = DP.GDPModel()
    single_level_to_gdp =
        MOI.copy_to(JuMP.backend(gdp_model), single_level_model)
    DP.reformulate_model(gdp_model, reformulating_solver.gdp_method)
    gdp_to_solver =
        MOI.copy_to(reformulating_solver.inner_solver, JuMP.backend(gdp_model))
    single_level_to_solver = MOI.Utilities.IndexMap()
    for single_level_index in keys(single_level_to_gdp)
        gdp_index = single_level_to_gdp[single_level_index]
        # the `DisjunctionSet` constraints are gone after reformulation
        if haskey(gdp_to_solver, gdp_index)
            single_level_to_solver[single_level_index] =
                gdp_to_solver[gdp_index]
        end
    end
    return single_level_to_solver
end

function MOI.is_empty(reformulating_solver::_ReformulatingSolver)
    return MOI.is_empty(reformulating_solver.inner_solver)
end

function MOI.empty!(reformulating_solver::_ReformulatingSolver)
    return MOI.empty!(reformulating_solver.inner_solver)
end

function MOI.optimize!(reformulating_solver::_ReformulatingSolver)
    return MOI.optimize!(reformulating_solver.inner_solver)
end

const _ModelOrOptimizerAttribute =
    Union{MOI.AbstractModelAttribute,MOI.AbstractOptimizerAttribute}

function MOI.supports(
    reformulating_solver::_ReformulatingSolver,
    attr::MOI.AnyAttribute,
)
    return MOI.supports(reformulating_solver.inner_solver, attr)
end

function MOI.supports(
    reformulating_solver::_ReformulatingSolver,
    attr::MOI.AnyAttribute,
    index_type::Type{<:MOI.Index},
)
    return MOI.supports(reformulating_solver.inner_solver, attr, index_type)
end

function MOI.get(
    reformulating_solver::_ReformulatingSolver,
    attr::_ModelOrOptimizerAttribute,
)
    return MOI.get(reformulating_solver.inner_solver, attr)
end

function MOI.get(
    reformulating_solver::_ReformulatingSolver,
    attr::MOI.AnyAttribute,
    index::Union{MOI.Index,Vector{<:MOI.Index}},
)
    return MOI.get(reformulating_solver.inner_solver, attr, index)
end

function MOI.set(
    reformulating_solver::_ReformulatingSolver,
    attr::_ModelOrOptimizerAttribute,
    value,
)
    return MOI.set(reformulating_solver.inner_solver, attr, value)
end

function MOI.set(
    reformulating_solver::_ReformulatingSolver,
    attr::MOI.AnyAttribute,
    index::MOI.Index,
    value,
)
    return MOI.set(reformulating_solver.inner_solver, attr, index, value)
end

function BilevelJuMP._mode_solver(
    solver::MOI.ModelLike,
    mode::BilevelJuMP.DisjunctionMode,
)
    return _ReformulatingSolver(solver, mode.gdp_method)
end

end # module
