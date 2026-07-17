# High-level simulation timing (LAMMPS-style report)
#
# Activated only by `simulate!(...; timing=true)`. Section timers wrap Molly
# categories: Pair, Specific1–4, General, Neigh, Constraints, Loggers (plus
# derived Other). When printing, Specific1–4 appear as 1-Body…4-Body under a
# Specific header row (no summed totals on the parent). Pair and General are
# section-only. Report columns: min/avg/max time [s] / %total, then Performance.
# min/avg/max are per `@sim_section` invocation (not MPI ranks).

const SIM_SPECIFIC_SECTIONS = (:Specific1, :Specific2, :Specific3, :Specific4)

mutable struct SectionStats
    total_ns::Int64
    min_ns::Int64
    max_ns::Int64
    count::Int64
end

SectionStats() = SectionStats(Int64(0), typemax(Int64), Int64(0), Int64(0))

mutable struct SimTimingState
    active::Bool
    t0_ns::UInt64
    section_ns::Dict{Symbol, SectionStats}
end

SimTimingState() = SimTimingState(false, zero(UInt64), Dict{Symbol, SectionStats}())

const SIM_TIMING = SimTimingState()

function reset_sim_timers!()
    empty!(SIM_TIMING.section_ns)
    return nothing
end

@inline function _add_sim_section!(section::Symbol, dt_ns::Integer)
    d = SIM_TIMING.section_ns
    dt = Int64(dt_ns)
    stats = get!(SectionStats, d, section)
    stats.total_ns += dt
    stats.count += Int64(1)
    stats.min_ns = min(stats.min_ns, dt)
    stats.max_ns = max(stats.max_ns, dt)
    return nothing
end

@inline function _section_stats(section::Symbol)
    return get(SIM_TIMING.section_ns, section, nothing)
end

@inline function _section_ns(section::Symbol)
    stats = _section_stats(section)
    return stats === nothing ? Int64(0) : stats.total_ns
end

@inline function _specific_ns()
    total = Int64(0)
    for s in SIM_SPECIFIC_SECTIONS
        total += _section_ns(s)
    end
    return total
end

# Sum of Specific1–4 totals (for Other residual / get_sim_timings); not printed
# on the Specific parent row.
function _specific_stats()
    total = Int64(0)
    count = Int64(0)
    min_ns = typemax(Int64)
    max_ns = Int64(0)
    for s in SIM_SPECIFIC_SECTIONS
        stats = _section_stats(s)
        stats === nothing && continue
        iszero(stats.count) && continue
        total += stats.total_ns
        count += stats.count
        min_ns = min(min_ns, stats.min_ns)
        max_ns = max(max_ns, stats.max_ns)
    end
    return SectionStats(total, count > 0 ? min_ns : typemax(Int64), max_ns, count)
end

# Print labels for Specific1–4 (internal symbols unchanged).
const SIM_SPECIFIC_PRINT_NAMES = ("1-Body", "2-Body", "3-Body", "4-Body")

"""
    @sim_section section expr

Accumulate wall time for a high-level simulation section when timing is active.
`section` is `:Pair`, `:Specific1`–`:Specific4`, `:General`, `:Neigh`,
`:Constraints`, or `:Loggers`.

`expr` is evaluated once. Timing is gated by a cheap runtime check of
`SIM_TIMING.active` (set by timing `simulate!`), not a compile-time no-op —
unlike `start_timing!(Val{false})` at the simulate! boundary. No
`try`/`finally`: if `expr` throws, the section is not recorded (acceptable for
this profiling use case). Hard crashes (segfault, `kill -9`) never run cleanup
either.
"""
macro sim_section(section, expr)
    # Runtime gate only: `active` is not known at expansion. Keep a single `expr`
    # body (no if/else duplication). Off-path cost is one Bool load + optional
    # zeroed `time_ns` placeholder — not a specialized compile-away no-op.
    return quote
        local _timing_on = SIM_TIMING.active
        local _t0 = _timing_on ? time_ns() : zero(UInt64)
        local _result = $(esc(expr))
        _timing_on && _add_sim_section!($(esc(section)), time_ns() - _t0)
        _result
    end
end

function get_sim_timings()
    out = Dict{String, Float64}(
        "Pair"        => _section_ns(:Pair) / 1e9,
        "Specific"    => _specific_ns() / 1e9,
        "General"     => _section_ns(:General) / 1e9,
        "Neigh"       => _section_ns(:Neigh) / 1e9,
        "Constraints" => _section_ns(:Constraints) / 1e9,
        "Loggers"     => _section_ns(:Loggers) / 1e9,
    )
    for name in SIM_SPECIFIC_SECTIONS
        out[String(name)] = _section_ns(name) / 1e9
    end
    return out
end

function _format_seconds(t::Real)
    t = Float64(t)
    iszero(t) && return "0"
    s = @sprintf("%.5f", t)
    s = rstrip(s, '0')
    return rstrip(s, '.')
end

_format_pct(pct::Real) = @sprintf("%.2f", Float64(pct))

# Unitful `dt` → ns; unitless `dt` treated as picoseconds (Molly NoUnits convention).
function _sim_dt_ns(dt)
    if dt isa Unitful.Quantity
        return Float64(ustrip(u"ns", dt))
    else
        return Float64(dt) * 1e-3 # ps → ns
    end
end

function _print_sim_timing_row(io::IO, name::AbstractString, stats::SectionStats, loop_s::Float64;
                               indent::Int=0, blank_extrema::Bool=false, blank_values::Bool=false)
    label = (" "^indent) * name
    if blank_values
        # Specific parent: section name only; numbers live on body sub-rows.
        @printf(io, "%-16s | %-12s | %-12s | %-12s | %6s\n",
                label, "-", "-", "-", "-")
        return nothing
    end
    total_s = stats.total_ns / 1e9
    pct = loop_s > 0 ? 100 * total_s / loop_s : 0.0
    if blank_extrema
        # Other: residual in avg/%total; min/max not meaningful.
        min_str, avg_str, max_str = "-", _format_seconds(total_s), "-"
    elseif iszero(stats.count)
        min_str = avg_str = max_str = _format_seconds(0.0)
    else
        min_str = _format_seconds(stats.min_ns / 1e9)
        avg_str = _format_seconds((stats.total_ns / stats.count) / 1e9)
        max_str = _format_seconds(stats.max_ns / 1e9)
    end
    @printf(io, "%-16s | %-12s | %-12s | %-12s | %6s\n",
            label, min_str, avg_str, max_str, _format_pct(pct))
    return nothing
end

function _print_specific_children(io::IO, loop_s::Float64)
    # Arity order 1–4; skip arities with no recorded time.
    for (name, label) in zip(SIM_SPECIFIC_SECTIONS, SIM_SPECIFIC_PRINT_NAMES)
        stats = _section_stats(name)
        stats === nothing && continue
        stats.total_ns > 0 || continue
        _print_sim_timing_row(io, label, stats, loop_s; indent=2)
    end
    return nothing
end

function _print_performance_line(io::IO, n_steps::Integer, loop_s::Float64, dt)
    steps_per_s = loop_s > 0 ? n_steps / loop_s : 0.0
    if dt === nothing
        @printf(io, "Performance: %.3f timesteps/s\n", steps_per_s)
        return nothing
    end
    sim_ns = n_steps * _sim_dt_ns(dt)
    if loop_s <= 0 || sim_ns <= 0
        @printf(io, "Performance: 0 ns/day, 0 hours/ns, %.3f timesteps/s\n", steps_per_s)
        return nothing
    end
    ns_per_day = sim_ns * 86400 / loop_s
    hours_per_ns = loop_s / (3600 * sim_ns)
    @printf(io, "Performance: %.3f ns/day, %.3f hours/ns, %.3f timesteps/s\n",
            ns_per_day, hours_per_ns, steps_per_s)
    return nothing
end

function sim_timing_report(sys, n_steps::Integer, loop_ns::Integer;
                           n_threads::Integer=1, n_procs::Integer=1, dt=nothing)
    io = IOBuffer()
    loop_s = loop_ns / 1e9
    n_atoms = if hasproperty(sys, :replicas) && !isempty(sys.replicas)
        length(first(sys.replicas))
    else
        length(sys)
    end
    n_replicas = hasproperty(sys, :n_replicas) ? sys.n_replicas : 1
    print(io, "Loop time of ", _format_seconds(loop_s), " on ", n_procs,
          " procs for ", n_steps, " steps with ", n_atoms, " atoms")
    n_replicas > 1 && print(io, " ($n_replicas replicas)")
    println(io)
    if n_threads > 1
        println(io, "(Molly n_threads = ", n_threads, ")")
    end
    _print_performance_line(io, n_steps, loop_s, dt)
    println(io)

    empty_stats = SectionStats()
    pair_stats = something(_section_stats(:Pair), empty_stats)
    specific_ns = _specific_ns()
    general_stats = something(_section_stats(:General), empty_stats)
    neigh_stats = something(_section_stats(:Neigh), empty_stats)
    constraints_stats = something(_section_stats(:Constraints), empty_stats)
    loggers_stats = something(_section_stats(:Loggers), empty_stats)
    accounted_ns = pair_stats.total_ns + specific_ns + general_stats.total_ns +
                   neigh_stats.total_ns + constraints_stats.total_ns + loggers_stats.total_ns
    other_ns = max(Int64(0), Int64(loop_ns) - accounted_ns)
    other_stats = SectionStats(other_ns, typemax(Int64), Int64(0), Int64(0))

    @printf(io, "%-16s | %-12s | %-12s | %-12s | %6s\n",
            "Section", "min time [s]", "avg time [s]", "max time [s]", "%total")
    println(io, "-"^70)

    _print_sim_timing_row(io, "Pair", pair_stats, loop_s)
    _print_sim_timing_row(io, "Specific", empty_stats, loop_s; blank_values=true)
    _print_specific_children(io, loop_s)
    _print_sim_timing_row(io, "General", general_stats, loop_s)
    _print_sim_timing_row(io, "Neigh", neigh_stats, loop_s)
    _print_sim_timing_row(io, "Constraints", constraints_stats, loop_s)
    _print_sim_timing_row(io, "Loggers", loggers_stats, loop_s)
    _print_sim_timing_row(io, "Other", other_stats, loop_s; blank_extrema=true)
    return String(take!(io))
end

function print_sim_timing_report(io::IO, sys, n_steps::Integer, loop_ns::Integer;
                                 n_threads::Integer=1, n_procs::Integer=1, dt=nothing)
    print(io, sim_timing_report(sys, n_steps, loop_ns; n_threads=n_threads, n_procs=n_procs, dt=dt))
    return nothing
end

# Manual start/stop for `simulate!` (avoids a closure when timing is off).
# Specialized on `Val` so the timing=false path is a compile-time no-op.
# `timing=true` starts a fresh session (stores wall-clock `t0_ns`) and prints
# on the success-path stop. Nested REMD replica steps use `timing=false` and
# must not see `active` (REMD clears it before spawning replicas); stop still
# uses `Val(timing)` so the wall-clock report prints from the stored `t0_ns`.
@inline start_timing!(::Val{false}) = false

@inline function start_timing!(::Val{true})
    # Reclaim any stale `active` left by a previous failed timing run that never
    # reached stop (no try/finally around simulate! bodies).
    SIM_TIMING.active = false
    reset_sim_timers!()
    SIM_TIMING.t0_ns = time_ns()
    SIM_TIMING.active = true
    return true
end

@inline start_timing!(timing::Bool) = start_timing!(Val(timing))

@inline stop_timing!(sys, n_steps::Integer, ::Val{false};
                     n_threads::Integer=1, io::IO=stdout, dt=nothing) = nothing

@inline function stop_timing!(sys, n_steps::Integer, ::Val{true};
                              n_threads::Integer=1, io::IO=stdout, dt=nothing)
    SIM_TIMING.active = false
    print_sim_timing_report(io, sys, n_steps, Int64(time_ns() - SIM_TIMING.t0_ns);
                            n_threads=n_threads, dt=dt)
    return nothing
end

@inline stop_timing!(sys, n_steps::Integer, timing::Bool; kwargs...) =
    stop_timing!(sys, n_steps, Val(timing); kwargs...)
