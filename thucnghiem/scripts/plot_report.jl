using Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using JSON3
using DataFrames
using Statistics
using CairoMakie
using CSV
using Printf

const ORDER = ["GA", "HGA", "NSGA2", "MO-HGA-LS", "Louvain", "Leiden"]
const LABELS = Dict(
    "GA" => "GA",
    "HGA" => "HGA (GA+LS)",
    "NSGA2" => "NSGA-II",
    "MO-HGA-LS" => "MO-HGA-LS",
    "Louvain" => "Louvain",
    "Leiden" => "Leiden",
)
const COLORS = Dict(
    "GA" => "#8c8c8c",
    "HGA" => "#e69f00",
    "NSGA2" => "#56b4e9",
    "MO-HGA-LS" => "#d55e00",
    "Louvain" => "#009e73",
    "Leiden" => "#cc79a7",
)
const GA_FAMILY = Set(["GA", "HGA", "NSGA2", "MO-HGA-LS"])

const METRIC_ALIASES = Dict(
    :q => [:q, :maxQ_Q],
    :phi => [:phi, :maxQ_Phi],
    :nmi => [:nmi, :maxQ_NMI],
    :ari => [:ari, :maxQ_ARI],
    :runtime => [:runtime, :time_s],
    :hv_final => [:hv_final],
    :n_eval => [:n_eval],
    :ls_time => [:ls_time, :time_ls],
    :archive_size => [:archive_size],
    :ls_calls => [:ls_calls],
    :ls_changed => [:ls_changed],
    :ls_improved => [:ls_improved],
    :k => [:k, :maxQ_K],
    :generations => [:generations],
    :seed => [:seed],
)

const METRIC_SPECS = [
    (:q, "Modularity Q", true),
    (:phi, "Conductance Φ", false),
    (:nmi, "NMI", true),
    (:ari, "ARI", true),
    (:runtime, "Runtime (s)", false),
    (:hv_final, "Hypervolume", true),
    (:n_eval, "Fitness evaluations", true),
    (:k, "Number of communities K", false),
]

resolve_path(path) = isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))
safe_name(x) = replace(string(x), r"[^A-Za-z0-9_.-]+" => "_")
algo_label(a) = get(LABELS, a, a)
algo_color(a) = get(COLORS, a, "#666666")

function to_string_dict(x)
    Dict{String,Any}(String(k) => v for (k, v) in pairs(x))
end

function parse_json_records(raw::String)
    try
        parsed = JSON3.read(raw)
        parsed isa AbstractVector || error("Expected a JSON array.")
        return [to_string_dict(row) for row in parsed]
    catch array_error
        rows = Dict{String,Any}[]
        for (line_number, line) in enumerate(eachline(IOBuffer(raw)))
            s = strip(line)
            isempty(s) && continue
            s in ("[", "]", ",") && continue
            startswith(s, "[") && (s = strip(s[2:end]))
            endswith(s, ",") && (s = strip(s[1:end-1]))
            endswith(s, "]") && (s = strip(s[1:end-1]))
            isempty(s) && continue
            try
                push!(rows, to_string_dict(JSON3.read(s)))
            catch err
                error("Cannot parse JSON at line $line_number: $(sprint(showerror, err))")
            end
        end
        isempty(rows) && error("No valid JSON records found. Array parser error: $(sprint(showerror, array_error))")
        return rows
    end
end

function load_runs(path::AbstractString)
    isfile(path) || error("Input file does not exist: $path")
    raw = strip(read(path, String))
    isempty(raw) && error("Input file is empty: $path")
    raw = replace(raw, r",\s*]" => "]")
    rows = parse_json_records(raw)
    isempty(rows) && error("Input file contains no records: $path")
    return DataFrame(rows)
end

function numeric_column!(df::DataFrame, col::Symbol)
    col in propertynames(df) || return df
    converted = Union{Missing,Float64}[]
    for x in df[!, col]
        if ismissing(x) || x === nothing
            push!(converted, missing)
        else
            y = tryparse(Float64, strip(string(x)))
            push!(converted, y === nothing || !isfinite(y) ? missing : y)
        end
    end
    df[!, col] = converted
    return df
end

function get_first_column(df::DataFrame, candidates::Vector{Symbol})
    for candidate in candidates
        candidate in propertynames(df) && return candidate
    end
    return nothing
end

function normalize_schema!(df::DataFrame)
    for required in (:dataset, :algo)
        required in propertynames(df) || error("Missing required field: $required")
        df[!, required] = String.(coalesce.(string.(df[!, required]), ""))
    end

    for (canonical, candidates) in METRIC_ALIASES
        source = get_first_column(df, candidates)
        if source !== nothing
            df[!, canonical] = df[!, source]
            numeric_column!(df, canonical)
        end
    end

    for col in propertynames(df)
        eltype(df[!, col]) <: Union{Missing,Real} && numeric_column!(df, col)
    end

    sortcols = [:algo, :dataset]
    :seed in propertynames(df) && push!(sortcols, :seed)
    sort!(df, sortcols)
    return df
end

function finite_values(df::DataFrame, dataset::AbstractString, metric::Symbol)
    metric in propertynames(df) || return Float64[]
    result = Float64[]
    for value in df[df.dataset .== dataset, metric]
        if !ismissing(value)
            x = try
                Float64(value)
            catch
                NaN
            end
            isfinite(x) && push!(result, x)
        end
    end
    return result
end

metric_values_by_dataset(df, dataset, metric) = finite_values(df, string(dataset), Symbol(metric))
mean_or_nan(values) = isempty(values) ? NaN : mean(values)
std_or_zero(values) = length(values) > 1 ? std(values) : 0.0

function algorithms_in(df::DataFrame)
    present = Set(String.(df.algo))
    result = [a for a in ORDER if a in present]
    append!(result, sort!([a for a in present if !(a in ORDER)]))
    return result
end

function save_figure(fig, path::AbstractString)
    mkpath(dirname(path))
    save(path, fig)
    println("  [OK] ", path)
    return path
end

function add_no_data!(ax, title_text)
    ax.title = title_text
    text!(ax, 0.5, 0.5;
        text="Không có dữ liệu",
        space=:relative,
        align=(:center, :center),
        color=:gray,
        fontsize=14,
    )
    return nothing
end

function set_dataset_ticks!(ax, datasets)
    ax.xticks = (1:length(datasets), datasets)
    ax.xticklabelrotation = pi / 7
    return nothing
end

function fig_algo_dashboard(sub, algo, figure_root)
    datasets = sort(unique(sub.dataset))
    fig = Figure(size=(1500, 900))
    panels = [
        (:q, "Modularity Q (↑)", false),
        (:phi, "Conductance Φ (↓)", false),
        (:nmi, "NMI (↑)", false),
        (:ari, "ARI (↑)", false),
        (:runtime, "Runtime (s, log)", true),
        (:k, "Number of communities K", false),
        (:hv_final, "Hypervolume (↑)", false),
        (:n_eval, "Fitness evaluations", false),
    ]
    for (i, (metric, title_text, logscale)) in enumerate(panels)
        ax = Axis(fig[(i - 1) ÷ 4 + 1, (i - 1) % 4 + 1])
        any_data = false
        for (d_idx, d) in enumerate(datasets)
            values = metric_values_by_dataset(sub, d, metric)
            if logscale
                values = filter(>(0), values)
            end
            isempty(values) && continue
            any_data = true
            n = length(values)
            offsets = [d_idx + 0.055 * sin(2pi * j / max(n, 1)) for j in 1:n]
            scatter!(ax, offsets, values; color=algo_color(algo), markersize=8)
            m = mean(values)
            scatter!(ax, [d_idx], [m]; color=algo_color(algo), marker=:diamond, markersize=15, strokecolor=:black, strokewidth=1.2)
        end
        if !any_data
            add_no_data!(ax, title_text)
        else
            ax.title = title_text
            set_dataset_ticks!(ax, datasets)
            logscale && (ax.yscale = log10)
        end
    end
    Label(fig[0, :], "Algorithm: $(algo_label(algo)) — Performance across datasets"; fontsize=21)
    out = joinpath(figure_root, safe_name(algo))
    save_figure(fig, joinpath(out, "dashboard.png"))
end

function fig_algo_metric_bars(sub, algo, figure_root)
    datasets = sort(unique(sub.dataset))
    fig = Figure(size=(1450, 470))
    metrics = [(:q, "Q"), (:phi, "Φ"), (:nmi, "NMI"), (:ari, "ARI")]
    for (i, (metric, title_text)) in enumerate(metrics)
        ax = Axis(fig[1, i])
        means = [mean_or_nan(metric_values_by_dataset(sub, d, metric)) for d in datasets]
        sds = [std_or_zero(metric_values_by_dataset(sub, d, metric)) for d in datasets]
        valid = [j for j in eachindex(datasets) if isfinite(means[j])]
        if isempty(valid)
            add_no_data!(ax, "$title_text — mean ± SD")
        else
            for j in valid
                barplot!(ax, [j], [means[j]]; color=algo_color(algo), width=0.65)
                errorbars!(ax, [j], [means[j]], [sds[j]]; whiskerwidth=8, color=:black)
                text!(ax, j, means[j] + sds[j]; text=@sprintf("%.4f", means[j]), align=(:center, :bottom), fontsize=10)
            end
            ax.title = "$title_text — mean ± SD"
            set_dataset_ticks!(ax, datasets)
        end
    end
    Label(fig[0, :], "Algorithm: $(algo_label(algo)) — Quality Metrics"; fontsize=20)
    out = joinpath(figure_root, safe_name(algo))
    save_figure(fig, joinpath(out, "bars_metrics.png"))
end

function per_algo_figures(df, algo, figure_root)
    sub = df[df.algo .== algo, :]
    out = joinpath(figure_root, safe_name(algo))
    mkpath(out)
    jobs = [
        ("dashboard", () -> fig_algo_dashboard(sub, algo, figure_root)),
        ("bars", () -> fig_algo_metric_bars(sub, algo, figure_root)),
    ]
    made = String[]
    for (name, job) in jobs
        try
            job()
            push!(made, name)
        catch err
            @warn "Could not create figure for algorithm" algo figure=name exception=(err, catch_backtrace())
        end
    end
    return made
end


function write_tables(df, table_root)
    mkpath(table_root)
    CSV.write(joinpath(table_root, "runs_normalized.csv"), df)

    summary = DataFrame(
        algo=String[], dataset=String[], runs=Int[],
    )
    metric_columns = Symbol[]
    for metric in keys(METRIC_ALIASES)
        if metric in propertynames(df)
            push!(metric_columns, Symbol(metric, "_mean"))
            push!(metric_columns, Symbol(metric, "_std"))
            summary[!, Symbol(metric, "_mean")] = Float64[]
            summary[!, Symbol(metric, "_std")] = Float64[]
        end
    end

    for algo in algorithms_in(df), dataset in sort(unique(df.dataset))
        sub = df[(df.algo .== algo) .& (df.dataset .== dataset), :]
        nrow(sub) == 0 && continue
        row = Dict{Symbol,Any}(:algo => algo, :dataset => dataset, :runs => nrow(sub))
        for metric in keys(METRIC_ALIASES)
            metric in propertynames(df) || continue
            values = finite_values(sub, dataset, metric)
            row[Symbol(metric, "_mean")] = mean_or_nan(values)
            row[Symbol(metric, "_std")] = std_or_zero(values)
        end
        push!(summary, row; cols=:union)
    end
    CSV.write(joinpath(table_root, "runs_summary.csv"), summary)
    return summary
end


function main()
    input_arg = length(ARGS) >= 1 ? ARGS[1] : "results/main/runs.jsonl"
    output_arg = length(ARGS) >= 2 ? ARGS[2] : "results/report"
    input_file = resolve_path(input_arg)
    output_dir = resolve_path(output_arg)
    figure_dir = joinpath(output_dir, "figures")
    table_dir = joinpath(output_dir, "tables")
    mkpath(figure_dir)
    mkpath(table_dir)

    df = load_runs(input_file)
    normalize_schema!(df)

    println("Loaded $(nrow(df)) runs from: $input_file")
    println("Algorithms: ", join(algorithms_in(df), ", "))
    println("Datasets  : ", join(sort(unique(df.dataset)), ", "))

    write_tables(df, table_dir)

    made = 0
    # Duyệt qua từng thuật toán thay vì Dataset để xuất thư mục thuật toán
    for algo in algorithms_in(df)
        made += length(per_algo_figures(df, algo, figure_dir))
    end

    println()
    println("="^68)
    println("REPORT GENERATION COMPLETED")
    println("="^68)
    println("Records     : ", nrow(df))
    println("Figures     : ", made)
    println("Figures dir : ", figure_dir)
    println("Tables dir  : ", table_dir)
    println("="^68)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end