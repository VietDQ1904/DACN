using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using MOHGALS
using Random
using JSON3
using CSV
using DataFrames
using Statistics
using Printf
using Base.Threads

function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    if i === nothing
        return default
    end
    return i < length(ARGS) ? ARGS[i + 1] : default
end

hasflag(flag) = flag in ARGS

function listarg(flag, default)
    x = argval(flag, nothing)
    return x === nothing ? default : split(x, ',')
end

parseint(x) = parse(Int, x)
parsefloat(x) = parse(Float64, x)


out = argval("--out", "results/main")
data_dir = argval("--data", "data")
datasets = listarg("--dataset", ["karate", "polbooks", "football", "email"])
algos = listarg("--algo", ["GA", "HGA", "NSGA2", "MO-HGA-LS", "Louvain", "Leiden"])

runs = parseint(argval("--runs", "30"))
seed0 = parseint(argval("--seed0", "0"))
seed_start = parseint(argval("--seed-start", string(seed0)))
seed_end_default = seed_start + runs - 1
seed_end = parseint(argval("--seed-end", string(seed_end_default)))
seed_end >= seed_start || error("--seed-end must be >= --seed-start")
seed_values = collect(seed_start:seed_end)

runs_actual = length(seed_values)


skip_leiden = hasflag("--skip-leiden")
use_cuda = hasflag("--cuda")
sequential = hasflag("--sequential")

# CUDA always runs sequentially because there is one GPU.
if use_cuda
    sequential = true
end


cfg = Config(
    mu = parseint(argval("--mu", "100")),
    lam = parseint(argval("--lam", "100")),
    pc = parsefloat(argval("--pc", "0.85")),
    pm = parsefloat(argval("--pm", "0")),
    tournament = parseint(argval("--tournament", "2")),
    i_ls = parseint(argval("--i_ls", "30")),
    p_ls = parsefloat(argval("--p_ls", "0.2")),
    rho = parsefloat(argval("--rho", "0.7")),
    a_max = parseint(argval("--a_max", "100")),
    g_max = parseint(argval("--g_max", "200")),
    k_stag = parseint(argval("--k_stag", "30")),
    eps_stop = parsefloat(argval("--eps_stop", "1e-4"))
)


if hasflag("--preset-fast")
    cfg.mu = 20
    cfg.lam = 20
    cfg.g_max = 30
    cfg.k_stag = 10
    cfg.i_ls = 3
    cfg.p_ls = 0.05
end

mkpath(out)


@info "Output directory" out
@info "Data directory" data_dir
@info "Datasets" datasets
@info "Algorithms" algos
@info "Seed range" "$(seed_start):$(seed_end)"
@info "Number of seeds" runs_actual
@info "Threads" Threads.nthreads()
@info "CUDA" use_cuda
@info "Sequential" sequential

graphs = Dict{String, GraphData}()

for d in datasets
    @info "Loading $d"
    graphs[d] = load_dataset(String(d), data_dir)
    g = graphs[d]
    @info(
        "$(d): " *
        "n=$(g.n), " *
        "m=$(g.m), " *
        "avgdeg=$(g.stats["avg_degree"]), " *
        "CC=$(g.stats["avg_cc"])"
    )
end


algos_julia = [
    a for a in algos if a != "Leiden"
]

tasks_julia = [
    (d, a, s) for d in datasets for a in algos_julia for s in seed_values
]

leiden_tasks = [
    (d, s) for d in datasets for s in seed_values
]

n_leiden_tasks = ("Leiden" in algos && !skip_leiden) ? length(leiden_tasks) : 0
total_tasks_count = length(tasks_julia) + n_leiden_tasks

local_results = Vector{Union{Nothing, RunResult}}(undef, total_tasks_count)
fill!(local_results, nothing)

@info "JIT warm-up..."

let
    g0 = load_dataset("karate", data_dir)
    c0 = Config(
        mu = 10,
        lam = 10,
        g_max = 2,
        k_stag = 2,
        i_ls = 2,
        p_ls = 0.5
    )

    for a in ("GA", "HGA", "NSGA2", "MO-HGA-LS")
        run_algorithm(g0, a, c0, 0)
    end
    run_baseline(g0, "Louvain", 0)
end

evaluators = Dict{String, Any}()

if use_cuda
    @info "Initializing CUDA..."
    Base.eval(Main, :(using CUDA))
    Base.include(Main, joinpath(@__DIR__, "..", "src", "cuda_backend.jl"))

    make_gpu_evaluator(g) =
        Base.invokelatest(() -> Main.MOHGALSCuda.GPUEvaluator(g))

    for d in datasets
        evaluators[String(d)] = make_gpu_evaluator(graphs[d])
    end
    @info(
        "CUDA evaluator enabled " *
        "(batch decode + Q/phi on GPU)"
    )

end

lk = ReentrantLock()

function exec_task(idx, d, a, s)
    g = graphs[d]

    try
        algo = String(a)

        if algo in ("GA", "HGA", "NSGA2", "MO-HGA-LS")
            if use_cuda
                local_results[idx] = Base.invokelatest(
                    run_algorithm, g, algo, cfg, s;
                    evaluator=evaluators[String(d)]
                )
            else
                local_results[idx] = run_algorithm(g, algo, cfg, s)
            end
        elseif algo == "Louvain"
            r = run_baseline(g, "Louvain", s)
            lab = r.lab
            local_results[idx] =
                RunResult(
                    d, a, s,
                    r.q, r.phi,
                    MOHGALS.truth_nmi(
                        g.truth,
                        lab
                    ),
                    MOHGALS.truth_ari(
                        g.truth,
                        lab
                    ),
                    maximum(lab), r.runtime, 0, 0,
                    1, NaN, Float64[], Float64[],
                    Float64[], 0, 0,
                    0, 0.0, 0.0, 0.0, lab
                )

        else
            error("Unsupported algorithm: $algo")
        end

        lock(lk) do
            @info @sprintf("done %s / %s / seed=%d", d, a, s)
        end
    catch e
        lock(lk) do
            @error(
                "failed $d $a seed=$s",
                exception = (
                    e,
                    catch_backtrace()
                )
            )
        end
    end
end

@info("Starting Julia algorithms (tasks=$(length(tasks_julia)), threads=$(Threads.nthreads()), sequential=$sequential, cuda=$use_cuda)")

if sequential
    for idx in eachindex(tasks_julia)
        exec_task(idx, tasks_julia[idx]...)
    end
else
    @threads for idx in eachindex(tasks_julia)
        exec_task(idx, tasks_julia[idx]...)
    end
end

if "Leiden" in algos && !skip_leiden
    @info("Running Leiden baselines sequentially on main thread...")
    offset = length(tasks_julia)

    for (i, (d, s)) in enumerate(leiden_tasks)
        g = graphs[d]
        idx = offset + i

        try
            r = run_baseline(g, "Leiden", s)
            lab = r.lab
            local_results[idx] =
                RunResult(
                    d, "Leiden",
                    s, r.q, r.phi,
                    MOHGALS.truth_nmi(
                        g.truth,
                        lab
                    ),
                    MOHGALS.truth_ari(
                        g.truth,
                        lab
                    ),
                    maximum(lab), r.runtime,
                    0, 0, 1,
                    NaN, Float64[], Float64[], Float64[],
                    0, 0, 0, 0.0, 0.0, 0.0, lab
                )

            @info @sprintf("done %s / Leiden / seed=%d", d, s)
        catch e
            @error(
                "failed $d Leiden seed=$s",
                exception = (
                    e,
                    catch_backtrace()
                )
            )
        end
    end
end

results = [x for x in local_results if x !== nothing]
runs_path = joinpath(out, "runs.jsonl")
save_runs(runs_path, results)

# Export the community assignment produced by every algorithm/run.
# One row in community_assignments.csv represents one vertex in one run.
assignment_rows = NamedTuple[]
community_rows = NamedTuple[]

for r in results
    g = graphs[r.dataset]
    graph_id = "$(g.name)_n$(g.n)_m$(g.m)"
    labels = MOHGALS.canonical(copy(r.labels))
    k = isempty(labels) ? 0 : maximum(labels)
    sizes = zeros(Int, k)
    degree_volumes = zeros(Int, k)
    internal_edges = zeros(Int, k)

    for u in 1:g.n
        c = labels[u]
        sizes[c] += 1
        degree_volumes[c] += g.deg[u]
        truth_label = g.truth === nothing || g.truth[u] < 0 ? missing : g.truth[u]
        push!(assignment_rows, (
            graph_id=graph_id,
            dataset=r.dataset,
            algorithm=r.algo,
            seed=r.seed,
            node_id=u,
            community_id=c,
            community_size=0, # filled after all community sizes are known
            truth_community=truth_label,
        ))
    end

    for e in eachindex(g.edges_u)
        u, v = g.edges_u[e], g.edges_v[e]
        if labels[u] == labels[v]
            internal_edges[labels[u]] += 1
        end
    end

    for c in 1:k
        cut_edges = degree_volumes[c] - 2 * internal_edges[c]
        denom = min(degree_volumes[c], 2 * g.m - degree_volumes[c])
        conductance_value = denom <= 0 ? 1.0 : cut_edges / denom
        push!(community_rows, (
            graph_id=graph_id,
            dataset=r.dataset,
            algorithm=r.algo,
            seed=r.seed,
            community_id=c,
            community_size=sizes[c],
            internal_edges=internal_edges[c],
            degree_volume=degree_volumes[c],
            conductance=conductance_value,
        ))
    end
end

if !isempty(assignment_rows)
    assignments = DataFrame(assignment_rows)
    # Fill each vertex row with the size of its assigned community.
    for row in eachrow(assignments)
        idx = findfirst(x -> x.graph_id == row.graph_id && x.dataset == row.dataset &&
            x.algorithm == row.algorithm && x.seed == row.seed &&
            x.community_id == row.community_id, community_rows)
        idx !== nothing && (row.community_size = community_rows[idx].community_size)
    end
    assignment_path = joinpath(out, "community_assignments.csv")
    CSV.write(assignment_path, assignments)

    community_path = joinpath(out, "community_summary.csv")
    CSV.write(community_path, DataFrame(community_rows))
    @info "Community assignments CSV" assignment_path
    @info "Community summary CSV" community_path
end

config_path = joinpath(out, "config.json")

open(config_path, "w") do io
    JSON3.write(io, Dict(
        "datasets" => datasets,
        "algos" => algos,
        "runs" => runs_actual,
        "seed0" => seed0,
        "seed_start" => seed_start,
        "seed_end" => seed_end,
        "seeds" => seed_values,
        "data_dir" => data_dir,
        "output_dir" => out,
        "sequential" => sequential,
        "threads" => Threads.nthreads(),
        "cuda" => use_cuda,
        "skip_leiden" => skip_leiden,
        "preset_fast" => hasflag("--preset-fast"),
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
        )
    ))
end

println()
println("=" ^ 70)
println("EXPERIMENT COMPLETED")
println("=" ^ 70)

println("Seeds      : ", seed_start, ":", seed_end, " (", runs_actual, " runs)")
println("Datasets   : ", join(datasets, ", "))
println("Algorithms : ", join(algos, ", "))
println("Results    : ", length(results), " / ", total_tasks_count)
println("Mode:", sequential ? "sequential" : "multi-threaded")
println("Threads:", Threads.nthreads())
println("CUDA:", use_cuda)
println("Runs file:", runs_path)
println("Config:", config_path)