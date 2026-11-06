# Per-subphase measurements inside local_search.
mutable struct LSProfile
    times_ns::Dict{Symbol,UInt64}
    bytes::Dict{Symbol,Int}
    calls::Dict{Symbol,Int}
end

LSProfile() = LSProfile(
    Dict{Symbol,UInt64}(),
    Dict{Symbol,Int}(),
    Dict{Symbol,Int}(),
)

# `do` syntax passes the closure as the first argument.
function profile_block!(f::F, p::LSProfile, key::Symbol) where {F}
    t0 = time_ns()
    result = nothing

    allocated = @allocated begin
        result = f()
    end

    elapsed = time_ns() - t0

    p.times_ns[key] = get(p.times_ns, key, UInt64(0)) + elapsed
    p.bytes[key] = get(p.bytes, key, 0) + allocated
    p.calls[key] = get(p.calls, key, 0) + 1

    return result
end

profile_block!(f::F, ::Nothing, ::Symbol) where {F} = f()

# ============================================================
# Phase-level profiler for MOHGALS
# ============================================================
# Measures only major algorithmic phases. It intentionally does
# not instrument small operations such as individual Q/Phi calls.
#
# Memory = cumulative bytes allocated by @allocated in that phase,
# NOT peak RSS/process memory.
# ============================================================

mutable struct ProfileStats
    time::Dict{String,Float64}
    bytes::Dict{String,Int}
    ls_calls::Int
    ls_time::Float64
    ls_bytes::Int
    ls_profile::LSProfile
end

ProfileStats() = ProfileStats(
    Dict{String,Float64}(),
    Dict{String,Int}(),
    0,
    0.0,
    0,
    LSProfile(),
)

@inline function _profile_record!(s::ProfileStats, phase::String,
                                  elapsed::Float64, bytes::Int)
    s.time[phase] = get(s.time, phase, 0.0) + elapsed
    s.bytes[phase] = get(s.bytes, phase, 0) + bytes
    return nothing
end

# IMPORTANT: do-block syntax in Julia passes the anonymous function
# as the FIRST argument. Therefore the signature must be (f, stats, phase),
# not (stats, phase, f).
function _profile_phase!(f::Function, s::ProfileStats, phase::String)
    t0 = time_ns()
    result = nothing

    bytes = @allocated begin
        result = f()
    end

    elapsed = (time_ns() - t0) / 1e9
    _profile_record!(s, phase, elapsed, bytes)
    return result
end

# Also allow the explicit form:
# _profile_phase!(stats, "phase", () -> work())
function _profile_phase!(s::ProfileStats, phase::String, f::Function)
    return _profile_phase!(f, s, phase)
end

const PROFILE_PHASES = (
    "initialization",
    "initial_fitness",
    "selection",
    "variation",
    "offspring_fitness",
    "local_search",
    "survivor_selection",
    "archive_update",
    "hypervolume_stopping",
    "final_metrics",
)

function _profile_percent(s::ProfileStats)
    total = get(s.time, "total", 0.0)
    Dict(
        k => total > 0 ? 100.0 * v / total : 0.0
        for (k, v) in s.time if k != "total"
    )
end

function profile_dict(s::ProfileStats)
    total_time = get(s.time, "total", 0.0)
    total_bytes = sum(values(s.bytes))

    out = Dict{String,Any}()

    for phase in PROFILE_PHASES
        t = get(s.time, phase, 0.0)
        b = get(s.bytes, phase, 0)
        out[phase] = Dict{String,Any}(
            "time_s" => t,
            "time_percent" => total_time > 0 ? 100.0 * t / total_time : 0.0,
            "allocated_bytes" => b,
            "allocated_mb" => b / 1024^2,
        )
    end

    out["local_search"]["calls"] = s.ls_calls
    out["local_search"]["avg_time_ms"] =
        s.ls_calls > 0 ? 1000.0 * s.ls_time / s.ls_calls : 0.0
    out["local_search"]["avg_allocated_mb"] =
        s.ls_calls > 0 ? s.ls_bytes / s.ls_calls / 1024^2 : 0.0

    ls_subphases = Dict{String,Any}()
    ls_total_ns = sum(values(s.ls_profile.times_ns); init=UInt64(0))

    for key in sort!(collect(keys(s.ls_profile.calls)); by=string)
        elapsed_ns = get(s.ls_profile.times_ns, key, UInt64(0))
        allocated = get(s.ls_profile.bytes, key, 0)
        calls = get(s.ls_profile.calls, key, 0)

        ls_subphases[string(key)] = Dict(
            "time_s" => elapsed_ns / 1e9,
            "time_percent_of_local_search" =>
                ls_total_ns > 0 ? 100.0 * elapsed_ns / ls_total_ns : 0.0,
            "allocated_bytes" => allocated,
            "allocated_mb" => allocated / 1024^2,
            "calls" => calls,
            "avg_time_ms" =>
                calls > 0 ? elapsed_ns / 1e6 / calls : 0.0,
            "avg_allocated_mb" =>
                calls > 0 ? allocated / calls / 1024^2 : 0.0,
        )
    end

    out["local_search"]["subphases"] = ls_subphases

    out["total"] = Dict(
        "time_s" => total_time,
        "allocated_bytes" => total_bytes,
        "allocated_mb" => total_bytes / 1024^2,
    )

    return out
end

"""
    run_algorithm_profile(g, algo, cfg, seed; evaluator=CPUEvaluator())

Same algorithmic behavior as `run_algorithm`, but records timing and
allocation for major phases. The profiler should not be used as the
official runtime benchmark because instrumentation itself has overhead.
"""
function run_algorithm_profile(
    g::GraphData,
    algo::String,
    cfg::Config,
    seed::Int;
    evaluator::Evaluator = CPUEvaluator(),
)
    @assert algo in ALGOS

    multi = algo in ("NSGA2", "MO-HGA-LS")
    use_ls = algo in ("HGA", "MO-HGA-LS") && cfg.p_ls > 0

    rng = MersenneTwister(seed)
    f = LARFactory(g, cfg.rho)
    pm = cfg.pm > 0 ? cfg.pm : 1 / g.n

    stats = ProfileStats()
    t_total = time_ns()
    lspool = use_ls ? make_ls_pool(g) : LSWorkspace[]

    # --------------------------------------------------------
    # Initialization + initial fitness
    # --------------------------------------------------------
    genes = _profile_phase!(stats, "initialization") do
        init_population(f, cfg.mu, rng)
    end

    labels, qs, ps = _profile_phase!(stats, "initial_fitness") do
        evaluate_batch(evaluator, g, genes)
    end
    n_eval = cfg.mu

    P = [F(qs[i], ps[i], multi) for i in 1:cfg.mu]
    rank, crowd = _profile_phase!(stats, "survivor_selection") do
        rank_crowding(P)
    end

    archive = _profile_phase!(stats, "archive_update") do
        archive_update(
            Tuple{Vector{Int},Float64,Float64}[],
            labels,
            qs,
            ps,
            multi,
            cfg.a_max,
        )
    end

    hv = _profile_phase!(stats, "hypervolume_stopping") do
        hypervolume([F(x[2], x[3], multi) for x in archive])
    end
    hvbest = hv
    stag = 0
    gen = 0

    hvh = [hv]
    qh = [maximum(x[2] for x in archive)]
    th = [(time_ns() - t_total) / 1e9]

    ls_calls = 0
    ls_changed = 0
    ls_improved = 0
    ls_time = 0.0
    ls_dq = 0.0
    ls_dp = 0.0

    # --------------------------------------------------------
    # Generation loop
    # --------------------------------------------------------
    while gen < cfg.g_max && stag < cfg.k_stag

        # Selection
        p1, p2 = _profile_phase!(stats, "selection") do
            p1_ = Vector{Int}(undef, cfg.lam)
            p2_ = similar(p1_)

            for arr in (p1_, p2_)
                for i in 1:cfg.lam
                    best = rand(rng, 1:cfg.mu)

                    for _ in 2:cfg.tournament
                        c = rand(rng, 1:cfg.mu)
                        better =
                            rank[c] < rank[best] ||
                            (rank[c] == rank[best] && crowd[c] > crowd[best])
                        better && (best = c)
                    end

                    arr[i] = best
                end
            end

            (p1_, p2_)
        end

        # Crossover + mutation
        offgenes = _profile_phase!(stats, "variation") do
            make_offspring(f, cfg, genes, p1, p2, pm, rng)
        end

        # Offspring fitness
        offlabels, oq, op = _profile_phase!(stats, "offspring_fitness") do
            evaluate_batch(evaluator, g, offgenes)
        end
        n_eval += cfg.lam

        # Local Search (parallel over the offspring batch)
        if use_ls
            t_ls = time_ns()
            st = nothing
            bytes_ls = @allocated begin
                st = ls_batch!(g, f, cfg, rng, multi, use_ls, lspool,
                               offgenes, offlabels, oq, op)
            end
            elapsed_ls = (time_ns() - t_ls) / 1e9
            stats.time["local_search"] = get(stats.time, "local_search", 0.0) + elapsed_ls
            stats.bytes["local_search"] = get(stats.bytes, "local_search", 0) + bytes_ls
            stats.ls_calls += st.calls
            stats.ls_time += elapsed_ls
            stats.ls_bytes += bytes_ls
            ls_calls += st.calls; ls_changed += st.changed; ls_improved += st.improved
            ls_dq += st.dq; ls_dp += st.dp
            ls_time += elapsed_ls
        end

        # Survivor pool + NSGA selection.
        # Keep exactly the same data flow as run_algorithm: no extra
        # rank/crowding recomputation is introduced by the profiler.
        genes, labels, qs, ps, rank, crowd = _profile_phase!(
            stats,
            "survivor_selection",
        ) do
            Rg = vcat(genes, offgenes)
            Rl = vcat(labels, offlabels)
            Rq = vcat(qs, oq)
            Rp = vcat(ps, op)

            keep = dedup_partition(Rl)

            if length(keep) < cfg.mu
                need = cfg.mu - length(keep)
                newg = init_population(f, need, rng)
                nl_, nq_, np_ = evaluate_batch(evaluator, g, newg)
                n_eval += need

                append!(Rg, newg)
                append!(Rl, nl_)
                append!(Rq, nq_)
                append!(Rp, np_)
                keep = collect(eachindex(Rg))
            end

            Rg = Rg[keep]
            Rl = Rl[keep]
            Rq = Rq[keep]
            Rp = Rp[keep]

            RF = [F(Rq[i], Rp[i], multi) for i in eachindex(Rq)]
            sel, rank_, crowd_ = nsga_select(RF, cfg.mu)

            (
                Rg[sel],
                Rl[sel],
                Rq[sel],
                Rp[sel],
                rank_,
                crowd_,
            )
        end

        # Archive update
        archive = _profile_phase!(stats, "archive_update") do
            archive_update(
                archive,
                offlabels,
                oq,
                op,
                multi,
                cfg.a_max,
            )
        end

        # Hypervolume + stopping
        hv = _profile_phase!(stats, "hypervolume_stopping") do
            hypervolume([F(x[2], x[3], multi) for x in archive])
        end

        if hv > hvbest + cfg.eps_stop
            hvbest = hv
            stag = 0
        else
            stag += 1
        end

        push!(hvh, hv)
        push!(qh, maximum(x[2] for x in archive))
        push!(th, (time_ns() - t_total) / 1e9)
        gen += 1
    end

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------
    chosen = _profile_phase!(stats, "final_metrics") do
        reps = select_representative(archive, multi)
        reps["maxQ"]
    end

    q, phi = chosen[2], chosen[3]
    lab = chosen[1]

    nmi, ari = _profile_phase!(stats, "final_metrics") do
        (truth_nmi(g.truth, lab), truth_ari(g.truth, lab))
    end

    stats.time["total"] = (time_ns() - t_total) / 1e9

    result = RunResult(
        g.name,
        algo,
        seed,
        q,
        phi,
        nmi,
        ari,
        maximum(lab),
        stats.time["total"],
        gen,
        n_eval,
        length(archive),
        hv,
        hvh,
        qh,
        th,
        ls_calls,
        ls_changed,
        ls_improved,
        ls_time,
        ls_dq,
        ls_dp,
        lab,
    )

    return result, stats
end

"""
    run_baseline_profile(g, algo, seed)

Profile Louvain/Leiden as a baseline. Baselines do not expose the same
internal evolutionary phases, so they are reported as one `baseline` phase.
"""
function run_baseline_profile(g::GraphData, algo::String, seed::Int)
    @assert algo in ("Louvain", "Leiden")

    stats = ProfileStats()
    t0 = time_ns()

    r = nothing
    bytes = @allocated begin
        r = run_baseline(g, algo, seed)
    end

    elapsed = (time_ns() - t0) / 1e9
    _profile_record!(stats, "baseline", elapsed, bytes)
    stats.time["total"] = elapsed

    lab = r.lab
    result = RunResult(
        g.name,
        algo,
        seed,
        r.q,
        r.phi,
        truth_nmi(g.truth, lab),
        truth_ari(g.truth, lab),
        maximum(lab),
        elapsed,
        0,
        0,
        1,
        NaN,
        Float64[],
        Float64[],
        Float64[],
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        lab,
    )

    return result, stats
end
