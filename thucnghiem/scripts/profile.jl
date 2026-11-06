using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MOHGALS
using JSON3
using CSV
using DataFrames
using Statistics
using Printf

# ============================================================
# Phase-level profiler
# ============================================================
# Examples:
#   julia --project=. scripts/profile.jl --dataset email --algo MO-HGA-LS --seed 1 --repeats 1
#   julia --project=. scripts/profile.jl --dataset football --algos GA,HGA,NSGA2,MO-HGA-LS --seed 1 --repeats 3
#   julia --project=. scripts/profile.jl --dataset football --algos Louvain,Leiden --seed 1 --repeats 3
#
# Evolutionary algorithms are profiled by major internal phases.
# Louvain/Leiden are reported as a single baseline phase because their
# internal implementation is external to the evolutionary engine.
#
# Memory = cumulative @allocated bytes, NOT peak RSS.
# ============================================================

function argval(args, flag, default=nothing)
    i = findfirst(==(flag), args)
    i === nothing ? default : (i < length(args) ? args[i + 1] : default)
end

function hasflag(args, flag)
    flag in args
end

function listarg(args, flag, default)
    x = argval(args, flag, nothing)
    x === nothing ? default : split(x, ',')
end

parseint(x) = parse(Int, x)
parsefloat(x) = parse(Float64, x)

function make_config(args)
    Config(
        mu = parseint(argval(args, "--mu", "100")),
        lam = parseint(argval(args, "--lam", "100")),
        pc = parsefloat(argval(args, "--pc", "0.85")),
        pm = parsefloat(argval(args, "--pm", "0")),
        tournament = parseint(argval(args, "--tournament", "2")),
        i_ls = parseint(argval(args, "--i_ls", "30")),
        p_ls = parsefloat(argval(args, "--p_ls", "0.2")),
        rho = parsefloat(argval(args, "--rho", "0.7")),
        a_max = parseint(argval(args, "--a_max", "100")),
        g_max = parseint(argval(args, "--g_max", "200")),
        k_stag = parseint(argval(args, "--k_stag", "30")),
        eps_stop = parsefloat(argval(args, "--eps_stop", "1e-4")),
    )
end

function phase_rows(stats_list)
    # Include baseline so the same output format can handle all algorithms.
    phases = [collect(MOHGALS.PROFILE_PHASES)..., "baseline"]
    rows = NamedTuple[]

    for phase in phases
        times = [get(s.time, phase, 0.0) for s in stats_list]
        bytes = [get(s.bytes, phase, 0) for s in stats_list]
        totals = [get(s.time, "total", 0.0) for s in stats_list]
        percents = [t > 0 ? 100.0 * x / t : 0.0 for (x, t) in zip(times, totals)]

        push!(rows, (
            phase = phase,
            time_mean_s = mean(times),
            time_std_s = length(times) > 1 ? std(times) : 0.0,
            time_min_s = minimum(times),
            time_max_s = maximum(times),
            time_percent_mean = mean(percents),
            allocated_mean_mb = mean(bytes) / 1024^2,
            allocated_std_mb = length(bytes) > 1 ? std(bytes) / 1024^2 : 0.0,
            allocated_min_mb = minimum(bytes) / 1024^2,
            allocated_max_mb = maximum(bytes) / 1024^2,
        ))
    end

    rows
end

function ls_subphase_rows(stats_list)
    phase_keys = sort!(
        unique(vcat(
            [collect(keys(s.ls_profile.calls)) for s in stats_list]...,
        ));
        by = string,
    )

    rows = NamedTuple[]
    for key in phase_keys
        times_s = [
            get(s.ls_profile.times_ns, key, UInt64(0)) / 1e9
            for s in stats_list
        ]
        alloc_mb = [
            get(s.ls_profile.bytes, key, 0) / 1024^2
            for s in stats_list
        ]
        calls = [
            get(s.ls_profile.calls, key, 0)
            for s in stats_list
        ]
        measured_ls_s = [
            sum(values(s.ls_profile.times_ns); init=UInt64(0)) / 1e9
            for s in stats_list
        ]
        percentages = [
            measured > 0 ? 100.0 * t / measured : 0.0
            for (t, measured) in zip(times_s, measured_ls_s)
        ]

        push!(rows, (
            subphase = string(key),
            time_mean_s = mean(times_s),
            time_percent_mean = mean(percentages),
            calls_mean = mean(calls),
            avg_time_ms = mean(calls) > 0 ?
                1000.0 * mean(times_s) / mean(calls) : 0.0,
            allocated_mean_mb = mean(alloc_mb),
            avg_allocated_mb = mean(calls) > 0 ?
                mean(alloc_mb) / mean(calls) : 0.0,
        ))
    end

    rows
end

function print_ls_subphase_table(stats_list)
    rows = ls_subphase_rows(stats_list)
    isempty(rows) && return

    println()
    println("LOCAL SEARCH SUBPHASE PROFILE")
    println("-"^118)
    @printf(
        "%-34s %11s %8s %10s %12s %13s %13s\n",
        "Subphase",
        "Time(s)",
        "%",
        "Calls",
        "Avg(ms)",
        "Alloc(MB)",
        "AvgAlloc(MB)",
    )
    println("-"^118)

    for r in rows
        @printf(
            "%-34s %11.6f %7.2f%% %10.2f %12.3f %13.3f %13.4f\n",
            r.subphase,
            r.time_mean_s,
            r.time_percent_mean,
            r.calls_mean,
            r.avg_time_ms,
            r.allocated_mean_mb,
            r.avg_allocated_mb,
        )
    end

    println("-"^118)
    println(
        "Note: subphase time percentages are normalized to the sum of ",
        "instrumented Local Search subphase times.",
    )
end

function print_phase_table(stats_list)
    rows = phase_rows(stats_list)

    println()
    println("PHASE PROFILE")
    println("-"^96)
    @printf("%-28s %12s %10s %14s\n", "Phase", "Time(s)", "%", "Alloc(MB)")
    println("-"^96)

    for r in rows
        r.time_mean_s == 0.0 && r.allocated_mean_mb == 0.0 && continue
        @printf(
            "%-28s %12.6f %9.2f%% %14.3f\n",
            r.phase,
            r.time_mean_s,
            r.time_percent_mean,
            r.allocated_mean_mb,
        )
    end

    total_time = [get(s.time, "total", 0.0) for s in stats_list]
    total_alloc = [sum(values(s.bytes)) for s in stats_list]

    println("-"^96)
    @printf(
        "%-28s %12.6f %9.2f%% %14.3f\n",
        "TOTAL",
        mean(total_time),
        100.0,
        mean(total_alloc) / 1024^2,
    )

    # Only meaningful for evolutionary algorithms.
    if any(s.ls_calls > 0 for s in stats_list)
        ls_calls = [s.ls_calls for s in stats_list]
        ls_time = [s.ls_time for s in stats_list]
        ls_bytes = [s.ls_bytes for s in stats_list]

        println()
        println("LOCAL SEARCH DETAIL")
        println("-"^64)
        @printf("Calls/run               : %.2f\n", mean(ls_calls))
        @printf("Time/run                : %.6f s\n", mean(ls_time))
        @printf(
            "Average time/call       : %.3f ms\n",
            mean(ls_calls) > 0 ? 1000.0 * mean(ls_time) / mean(ls_calls) : 0.0,
        )
        @printf("Allocated/run            : %.3f MB\n", mean(ls_bytes) / 1024^2)
        @printf(
            "Average allocation/call  : %.3f MB\n",
            mean(ls_calls) > 0 ? mean(ls_bytes) / mean(ls_calls) / 1024^2 : 0.0,
        )
    end
end

function main(args)
    data_dir = argval(args, "--data", "data")
    dataset_name = argval(args, "--dataset", "karate")

    # Support both --algo X and --algos A,B,C.
    single_algo = argval(args, "--algo", nothing)
    algos = single_algo === nothing ?
        listarg(args, "--algos", ["GA", "HGA", "NSGA2", "MO-HGA-LS"]) :
        [single_algo]

    seed_start = parseint(argval(args, "--seed", "1"))
    repeats = parseint(argval(args, "--repeats", "3"))
    out = argval(args, "--out", "results/profile")

    repeats >= 1 || error("--repeats must be >= 1")

    supported = Set(vcat(collect(ALGOS), ["Louvain", "Leiden"]))
    unknown = [a for a in algos if !(a in supported)]
    isempty(unknown) || error("Unsupported algorithms: $(join(unknown, ", "))")

    mkpath(out)
    cfg = make_config(args)

    if hasflag(args, "--preset-fast")
        cfg.mu = 50
        cfg.lam = 50
        cfg.g_max = 100
        cfg.k_stag = 20
        cfg.i_ls = 5
        cfg.p_ls = 0.1
    end

    println("="^96)
    println("MO-HGA-LS PHASE PROFILER")
    println("="^96)
    println("Dataset    : ", dataset_name)
    println("Algorithms : ", join(algos, ", "))
    println("Seed start : ", seed_start)
    println("Repeats    : ", repeats)
    println("Output     : ", out)
    println()

    # Dataset
    println("[1/3] Loading dataset...")
    load_t0 = time_ns()
    g = load_dataset(dataset_name, data_dir)
    load_time = (time_ns() - load_t0) / 1e9

    @printf(
        "n=%d, m=%d, avg_degree=%.3f, load=%.6f s\n",
        g.n,
        g.m,
        g.stats["avg_degree"],
        load_time,
    )

    # Warm-up JIT
    println()
    println("[2/3] Warming up JIT...")

    warm_cfg = Config(
        mu = min(10, cfg.mu),
        lam = min(10, cfg.lam),
        pc = cfg.pc,
        pm = cfg.pm,
        tournament = cfg.tournament,
        i_ls = min(2, cfg.i_ls),
        p_ls = cfg.p_ls,
        rho = cfg.rho,
        a_max = min(20, cfg.a_max),
        g_max = 2,
        k_stag = 2,
        eps_stop = cfg.eps_stop,
    )

    for algo in algos
        if algo in ALGOS
            run_algorithm(g, algo, warm_cfg, seed_start)
        else
            run_baseline(g, algo, seed_start)
        end
        println("  warm-up: ", algo)
    end

    # Profiling
    println()
    println("[3/3] Profiling...")
    println()

    all_summary_rows = NamedTuple[]

    for algo in algos
        println()
        println("="^96)
        println("ALGORITHM: ", algo)
        println("="^96)

        stats_list = ProfileStats[]
        results = RunResult[]

        for r in 1:repeats
            seed = seed_start + r - 1
            GC.gc()

            result, stats = if algo in ALGOS
                run_algorithm_profile(g, algo, cfg, seed)
            else
                run_baseline_profile(g, algo, seed)
            end

            push!(results, result)
            push!(stats_list, stats)

            @printf(
                "run=%d seed=%d total=%.6f s alloc=%.3f MB ls_calls=%d\n",
                r,
                seed,
                get(stats.time, "total", 0.0),
                sum(values(stats.bytes)) / 1024^2,
                stats.ls_calls,
            )
        end

        print_phase_table(stats_list)
        print_ls_subphase_table(stats_list)

        algo_file = replace(algo, "-" => "_")

        # ----------------------------------------------------
        # Phase CSV
        # ----------------------------------------------------
        CSV.write(
            joinpath(out, "$(dataset_name)_$(algo_file)_phases.csv"),
            DataFrame(phase_rows(stats_list)),
        )

        ls_rows = ls_subphase_rows(stats_list)
        if !isempty(ls_rows)
            CSV.write(
                joinpath(out, "$(dataset_name)_$(algo_file)_ls_subphases.csv"),
                DataFrame(ls_rows),
            )
        end

        # Per-run JSON
        profile_runs = [
            Dict(
                "dataset" => dataset_name,
                "algorithm" => algo,
                "seed" => seed_start + i - 1,
                "profile" => profile_dict(stats_list[i]),
                "q" => results[i].q,
                "phi" => results[i].phi,
                "nmi" => results[i].nmi,
                "ari" => results[i].ari,
                "generations" => results[i].generations,
                "n_eval" => results[i].n_eval,
                "archive_size" => results[i].archive_size,
                "ls_calls" => results[i].ls_calls,
                "ls_changed" => results[i].ls_changed,
                "ls_improved" => results[i].ls_improved,
            )
            for i in eachindex(stats_list)
        ]

        open(joinpath(out, "$(dataset_name)_$(algo_file)_profile.json"), "w") do io
            JSON3.write(io, profile_runs)
        end

        total_times = [get(s.time, "total", 0.0) for s in stats_list]
        total_alloc = [sum(values(s.bytes)) for s in stats_list]
        ls_times = [s.ls_time for s in stats_list]
        ls_calls = [s.ls_calls for s in stats_list]

        push!(all_summary_rows, (
            dataset = dataset_name,
            algorithm = algo,
            repeats = repeats,
            time_mean_s = mean(total_times),
            time_std_s = length(total_times) > 1 ? std(total_times) : 0.0,
            memory_alloc_mean_mb = mean(total_alloc) / 1024^2,
            memory_alloc_std_mb = length(total_alloc) > 1 ? std(total_alloc) / 1024^2 : 0.0,
            ls_time_mean_s = mean(ls_times),
            ls_calls_mean = mean(ls_calls),
            q_mean = mean(r.q for r in results),
            phi_mean = mean(r.phi for r in results),
        ))
    end

    # Combined summary + config
    summary_df = DataFrame(all_summary_rows)
    CSV.write(joinpath(out, "$(dataset_name)_profile_summary.csv"), summary_df)

    config_out = Dict(
        "dataset" => dataset_name,
        "data_dir" => data_dir,
        "algorithms" => algos,
        "seed_start" => seed_start,
        "repeats" => repeats,
        "config" => Dict(
            "mu" => cfg.mu,
            "lam" => cfg.lam,
            "pc" => cfg.pc,
            "pm" => cfg.pm,
            "tournament" => cfg.tournament,
            "i_ls" => cfg.i_ls,
            "p_ls" => cfg.p_ls,
            "rho" => cfg.rho,
            "a_max" => cfg.a_max,
            "g_max" => cfg.g_max,
            "k_stag" => cfg.k_stag,
            "eps_stop" => cfg.eps_stop,
        ),
    )

    open(joinpath(out, "profile_config.json"), "w") do io
        JSON3.write(io, config_out)
    end

    println()
    println("="^96)
    println("PROFILE COMPLETE")
    println("="^96)
    display(summary_df)
    println()
    println("Output directory: ", out)
end

main(ARGS)


