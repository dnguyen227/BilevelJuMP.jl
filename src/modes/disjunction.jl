# Copyright (c) 2019: Joaquim Dias Garcia, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    DisjunctionMode(gdp_method)

Used to solve a bilevel problem with the MPEC reformulation, writing each
complementarity condition as a two-term disjunction
`[dual == 0] ∨ [primal constraint == 0]` and reformulating the
disjunctions with DisjunctiveProgramming.jl before the solve.

* `gdp_method` is the DisjunctiveProgramming.jl reformulation, e.g.
  `BigM()`, `Hull()`, `MBM(optimizer)` or `Indicator()`.

The variable bounds the reformulation needs come from the bounds of the
model and from the bound hints (e.g. `set_dual_upper_bound_hint`).
Requires `using DisjunctiveProgramming`.
"""
mutable struct DisjunctionMode{T} <: AbstractBilevelSolverMode{T}
    gdp_method::Any
    function DisjunctionMode(gdp_method)
        return new{Float64}(gdp_method)
    end
end

# `DisjunctionSet` with the given disjunct sets; defined by the
# DisjunctiveProgramming extension
function _disjunction_set end

# Complementarity `dual * primal_func == 0` as the disjunction
# `[dual == 0] ∨ [primal_func == 0]`, selected by one binary
function add_complement(
    mode::DisjunctionMode{T},
    model,
    complement::Complement,
    primal_index_map,
    dual_index_map,
    copy_names::Bool,
    pass_start::Bool,
) where {T}
    if complement.is_vec
        error(
            "Vector constraint is (currently) not supported by DisjunctionMode",
        )
    end
    extension = :BilevelJuMPDisjunctiveProgrammingExt
    if Base.get_extension(@__MODULE__, extension) === nothing
        error("`DisjunctionMode` requires `using DisjunctiveProgramming`.")
    end

    primal_func =
        MOIU.map_indices(vref -> primal_index_map[vref], complement.func_w_cte)
    dual_vref = dual_index_map[complement.variable]

    # bvref = 1 selects [dual == 0], bvref = 0 selects [primal_func == 0]
    bvref, _ = MOI.add_constrained_variable(model, MOI.ZeroOne())
    activation = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{T}[], one(T))
    complement_indicator = MOIU.operate(-, T, one(T), bvref)
    disjunction_func = MOIU.operate(
        vcat,
        T,
        activation,
        bvref,
        complement_indicator,
        dual_vref,
        primal_func,
    )
    disjunct_sets = [[MOI.EqualTo(zero(T))], [MOI.EqualTo(zero(T))]]
    disjunction_con = MOI.add_constraint(
        model,
        disjunction_func,
        _disjunction_set(disjunct_sets),
    )

    if pass_start
        primal_start = MOIU.eval_variables(
            vref -> nothing_to_nan(
                MOI.get(model, MOI.VariablePrimalStart(), vref),
            ),
            primal_func,
        )
        if !isnan(primal_start)
            # a tight constraint may carry a nonzero dual: start in
            # [primal_func == 0]
            is_tight = abs(primal_start) < 1e-8
            MOI.set(
                model,
                MOI.VariablePrimalStart(),
                bvref,
                ifelse(is_tight, 0.0, 1.0),
            )
        end
    end

    if copy_names
        dual_name = MOI.get(model, MOI.VariableName(), dual_vref)
        MOI.set(model, MOI.VariableName(), bvref, "compl_bin_($(dual_name))")
        MOI.set(
            model,
            MOI.ConstraintName(),
            disjunction_con,
            "compl_disj_($(dual_name))",
        )
    end
    return disjunction_con
end
