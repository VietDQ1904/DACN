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
    :ls_delta_q => [
        :ls_delta_q,
        :sum_delta_q,
        :ls_sum_delta_q,
        :total_delta_q
    ],

    :ls_delta_phi => [
        :ls_delta_phi,
        :sum_delta_phi,
        :ls_sum_delta_phi,
        :total_delta_phi
    ],
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

function ls_bar_color(metric::Symbol, algo::AbstractString)

    palette = Dict(
        :ls_calls => Dict(
            "HGA"       => "#9ecae1",
            "MO-HGA-LS" => "#2171b5"
        ),

        :ls_improved => Dict(
            "HGA"       => "#a1d99b",
            "MO-HGA-LS" => "#238b45"
        ),

        :ls_dq => Dict(
            "HGA"       => "#9ecae1",
            "MO-HGA-LS" => "#2171b5"
        ),

        :ls_dphi => Dict(
            "HGA"       => "#fcae91",
            "MO-HGA-LS" => "#cb181d"
        )
    )

    metric_palette = get(palette, metric, nothing)

    isnothing(metric_palette) && return :gray

    return get(metric_palette, String(algo), :gray)
end


function ls_metric_mean(
    sub::DataFrame,
    dataset::AbstractString,
    metric::Symbol
)
    metric in propertynames(sub) || return NaN

    values = finite_values(sub, dataset, metric)
    return mean_or_nan(values)
end

function ls_add_bar_label!(
    ax,
    x::Real,
    y::Real;
    digits::Int = 1,
    color = :black,
    fontsize::Int = 9
)
    isfinite(y) || return

    label = @sprintf("%.*f", digits, y)

    offset = max(abs(y) * 0.012, 0.015)

    text!(
        ax,
        x,
        y >= 0 ? y + offset : y - offset;
        text = label,
        align = (:center, y >= 0 ? :bottom : :top),
        fontsize = fontsize,
        color = color
    )
end

function ls_add_point_label!(
    ax,
    x::Real,
    y::Real;
    digits::Int = 3,
    color = :black,
    fontsize::Int = 9
)
    isfinite(y) || return

    label = @sprintf("%.*f", digits, y)

    offset = max(abs(y) * 0.025, 0.015)

    text!(
        ax,
        x,
        y + offset;
        text = label,
        align = (:center, :bottom),
        fontsize = fontsize,
        color = color
    )
end


function add_ls_activity_panel!(
    ax,
    df::DataFrame,
    datasets::Vector{String}
)
    algos = ["HGA", "MO-HGA-LS"]

    metrics = [
        (:ls_calls, "Calls"),
        (:ls_improved, "Improvements")
    ]

    metric_centers = [-0.22, 0.22]
    algo_offsets = [-0.075, 0.075]

    bar_width = 0.14
    has_data = false

    for (dataset_idx, dataset) in enumerate(datasets)
        for (metric_idx, (metric, _)) in enumerate(metrics)
            center = dataset_idx + metric_centers[metric_idx]

            for (algo_idx, algo) in enumerate(algos)
                algo_data = df[df.algo .== algo, :]

                value = ls_metric_mean(
                    algo_data,
                    dataset,
                    metric
                )

                isfinite(value) || continue

                has_data = true

                x = center + algo_offsets[algo_idx]
                color = ls_bar_color(metric, algo)

                barplot!(
                    ax,
                    [x],
                    [value];
                    width = bar_width,
                    color = color,
                    strokecolor = :transparent
                )

                ls_add_bar_label!(
                    ax,
                    x,
                    value;
                    digits = 1,
                    color = color,
                    fontsize = 9
                )
            end
        end
    end

    if !has_data
        add_no_data!(ax, "No Local Search activity data")
    end

    return has_data
end

function add_ls_objective_panel!(
    fig,
    position,
    df::DataFrame,
    datasets::Vector{String}
)
    algos = ["HGA", "MO-HGA-LS"]

    ax = Axis(
        fig[position...],
        title = "(b) Objective Variation and LS Cost",
        xlabel = "Dataset",
        ylabel = "Cumulative objective variation",
        xticks = (1:length(datasets), datasets)
    )

    ax.titlesize = 15
    ax.xlabelsize = 12
    ax.ylabelsize = 12
    ax.xticklabelsize = 11
    ax.yticklabelsize = 11

    ax.xgridvisible = false
    ax.ygridvisible = true
    ax.ygridcolor = (:gray, 0.22)
    ax.ygridwidth = 0.8

    metrics = [
        :ls_dq,
        :ls_dphi
    ]

    metric_centers = [-0.20, 0.20]
    algo_offsets = [-0.075, 0.075]

    bar_width = 0.14
    has_bar_data = false

    all_objective_values = Float64[]

    for (dataset_idx, dataset) in enumerate(datasets)
        for (metric_idx, metric) in enumerate(metrics)
            center = dataset_idx + metric_centers[metric_idx]

            for (algo_idx, algo) in enumerate(algos)
                algo_data = df[df.algo .== algo, :]

                value = ls_metric_mean(
                    algo_data,
                    dataset,
                    metric
                )

                isfinite(value) || continue

                has_bar_data = true
                push!(all_objective_values, value)

                x = center + algo_offsets[algo_idx]
                color = ls_bar_color(metric, algo)

                barplot!(
                    ax,
                    [x],
                    [value];
                    width = bar_width,
                    color = color,
                    strokecolor = :transparent
                )

                ls_add_bar_label!(
                    ax,
                    x,
                    value;
                    digits = 2,
                    color = color,
                    fontsize = 9
                )
            end
        end
    end

    if !has_bar_data
        add_no_data!(ax, "No objective variation data")
    end

    if !isempty(all_objective_values)
        ymin = min(0.0, minimum(all_objective_values))
        ymax = max(0.0, maximum(all_objective_values))

        margin = max((ymax - ymin) * 0.15, 1.0)

        ylims!(ax, ymin - margin, ymax + margin)
    end

    axr = Axis(
        fig[position...],
        yaxisposition = :right,
        ylabel = "Local Search time (seconds)",
        backgroundcolor = :transparent
    )

    axr.ylabelsize = 12
    axr.yticklabelsize = 11

    axr.xgridvisible = false
    axr.ygridvisible = false

    axr.xticksvisible = false
    axr.xticklabelsvisible = false
    axr.xlabelvisible = false

    axr.leftspinevisible = false
    axr.bottomspinevisible = false
    axr.topspinevisible = false
    axr.rightspinevisible = true

    # Thu thập thời gian trung bình.
    all_time_values = Float64[]

    time_data = Dict{String, Vector{Float64}}()

    for algo in algos
        algo_data = df[df.algo .== algo, :]

        values = [
            ls_metric_mean(
                algo_data,
                dataset,
                :ls_time
            )
            for dataset in datasets
        ]

        time_data[algo] = values

        append!(
            all_time_values,
            filter(isfinite, values)
        )
    end

    if !isempty(all_time_values)
        ymax_time = maximum(all_time_values)
        ylims!(
            axr,
            0,
            ymax_time > 0 ? ymax_time * 1.30 : 1.0
        )
    end

    for algo in algos
        values = time_data[algo]

        valid_indices = findall(isfinite, values)

        isempty(valid_indices) && continue

        color = algo_color(algo)

        lines!(
            axr,
            valid_indices,
            values[valid_indices];
            color = color,
            linewidth = 2.5
        )

        scatter!(
            axr,
            valid_indices,
            values[valid_indices];
            color = color,
            markersize = 10
        )

        for idx in valid_indices
            ls_add_point_label!(
                axr,
                idx,
                values[idx];
                digits = 3,
                color = color,
                fontsize = 9
            )
        end
    end

    return ax, axr
end

function fig_runtime_comparison(
    df::DataFrame,
    figure_root::AbstractString
)
    algos = ORDER

    preferred_order = [
        "karate",
        "polbooks",
        "football",
        "email"
    ]

    available_datasets = Set(lowercase.(String.(df.dataset)))

    datasets = filter(
        d -> d in available_datasets,
        preferred_order
    )

    isempty(datasets) && begin
        @warn "No datasets available for runtime comparison."
        return false
    end

    runtime_stats = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    all_lower = Float64[]
    all_upper = Float64[]

    for dataset in datasets
        for algo in algos
            sub = df[df.algo .== algo, :]
            values = finite_values(sub, dataset, :runtime)

            filter!(>(0), values)
            isempty(values) && continue

            μ = mean(values)
            σ = std_or_zero(values)

            runtime_stats[(dataset, algo)] = (μ, σ)

            lower = max(μ - σ, μ * 0.05)
            upper = μ + σ
            push!(all_lower, lower)
            push!(all_upper, upper)
        end
    end

    isempty(all_lower) && begin
        @warn "No valid runtime data."
        return false
    end

    fig = Figure(
        size = (1650, 850),
        fontsize = 13,
        backgroundcolor = :white
    )

    ax = Axis(
        fig[1, 1],
        title = "(a) Runtime Comparison Across Algorithms",
        xlabel = "Dataset",
        ylabel = "Runtime (seconds, logarithmic scale)",
        yscale = log10
    )

    ax.titlesize = 17
    ax.xlabelsize = 13
    ax.ylabelsize = 13
    ax.xticklabelsize = 12
    ax.yticklabelsize = 11

    ax.xgridvisible = true
    ax.xgridcolor = (:gray, 0.22)
    ax.ygridvisible = true
    ax.ygridcolor = (:gray, 0.22)
    ax.ygridwidth = 0.8

    n_algos = length(algos)
    
    group_width = 0.78
    bar_width = group_width / n_algos
    offsets = range(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2;
        length = n_algos
    )

    for (dataset_idx, dataset) in enumerate(datasets)
        for (algo_idx, algo) in enumerate(algos)

            key = (dataset, algo)
            haskey(runtime_stats, key) || continue

            μ, σ = runtime_stats[key]
            x = dataset_idx + offsets[algo_idx]
            color = algo_color(algo)

            lower = max(μ - σ, μ * 0.05)
            upper = μ + σ

            barplot!(
                ax,
                [x],
                [μ];
                width = bar_width * 0.88,
                color = color
            )

            rangebars!(
                ax,
                [x],
                [lower],
                [upper];
                color = :black,
                linewidth = 1.6,
                whiskerwidth = 7
            )

            mean_str = @sprintf("%.3f", μ)
            delta_str = @sprintf("±%.3f", σ)

            text!(
                ax,
                x,
                upper * 1.12;
                text = mean_str,
                align = (:center, :bottom),
                fontsize = 9.5,
                color = :black
            )

            text!(
                ax,
                x + bar_width * 0.14,
                μ;
                text = delta_str,
                align = (:left, :center),
                fontsize = 8.5,
                color = :black
            )
        end
    end

    ax.xticks = (1:length(datasets), datasets)

    ymin = minimum(all_lower) * 0.4
    ymax = maximum(all_upper) * 2.5
    ylims!(ax, ymin, ymax)

    legend_elements = [
        PolyElement(color = algo_color(algo))
        for algo in algos
    ]

    legend_labels = [
        algo_label(algo)
        for algo in algos
    ]

    Legend(
        fig[2, 1],
        legend_elements,
        legend_labels;
        orientation = :horizontal,
        framevisible = false,
        tellwidth = false,
        tellheight = true,
        halign = :center,
        labelsize = 12,
        patchsize = (16, 12),
        colgap = 25
    )

    rowgap!(fig.layout, 20)

    out = joinpath(figure_root, "runtime")
    mkpath(out)

    save_figure(
        fig,
        joinpath(out, "runtime_comparison.png")
    )

    return true
end

function fig_ls_behavior(
    df::DataFrame,
    figure_root::AbstractString
)
    selected_algos = ["HGA", "MO-HGA-LS"]

    sub = df[in.(df.algo, Ref(selected_algos)), :]

    if isempty(sub)
        @warn "No HGA or MO-HGA-LS records available for Local Search analysis."
        return false
    end

    preferred_order = [
        "karate",
        "polbooks",
        "football",
        "email"
    ]

    available_datasets = Set(lowercase.(String.(sub.dataset)))

    datasets = filter(
        d -> d in available_datasets,
        preferred_order
    )

    if isempty(datasets)
        @warn "No recognized datasets found."
        return false
    end

    required_metrics = [
        :ls_calls,
        :ls_improved,
        :ls_dq,
        :ls_dphi,
        :ls_time
    ]

    missing_metrics = filter(
        metric -> !(metric in propertynames(sub)),
        required_metrics
    )

    if !isempty(missing_metrics)
        @warn "Missing Local Search fields in JSONL" missing_metrics
    end

    fig = Figure(
        size = (1950, 1200),
        fontsize = 13,
        backgroundcolor = :white
    )

    # Label(
    #     fig[1, 1:2],
    #     "LOCAL SEARCH BEHAVIOR ANALYSIS",
    #     fontsize = 23,
    #     font = :bold
    # )

    ax1 = Axis(
        fig[3, 1],
        title = "(a) Local Search Activity",
        xlabel = "Dataset",
        ylabel = "Number of operations",
        xticks = (1:length(datasets), datasets)
    )

    ax1.titlesize = 15
    ax1.xlabelsize = 12
    ax1.ylabelsize = 12
    ax1.xticklabelsize = 11
    ax1.yticklabelsize = 11

    ax1.xgridvisible = false
    ax1.ygridvisible = true
    ax1.ygridcolor = (:gray, 0.22)
    ax1.ygridwidth = 0.8

    add_ls_activity_panel!(
        ax1,
        sub,
        datasets
    )

    ax2, ax2r = add_ls_objective_panel!(
        fig,
        (3, 2),
        sub,
        datasets
    )

    algorithm_elements = [
        PolyElement(
            color = "#9ecae1",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#2171b5",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#a1d99b",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#238b45",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#fcae91",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#cb181d",
            strokecolor = :transparent
        )
    ]

    algorithm_labels = [
        "HGA — Calls",
        "MO-HGA-LS — Calls",
        "HGA — Improvements",
        "MO-HGA-LS — Improvements",
        "HGA — Objective Δ",
        "MO-HGA-LS — Objective Δ"
    ]

    Label(
        fig[4, 1:2],
        "ALGORITHM",
        fontsize = 12,
        font = :bold,
        color = :gray25,
        halign = :center
    )

    Legend(
        fig[5, 1:2],
        algorithm_elements,
        algorithm_labels;
        orientation = :horizontal,
        framevisible = false,
        tellwidth = false,
        tellheight = true,
        halign = :center,
        labelsize = 11,
        patchsize = (22, 14),
        padding = (10, 5, 10, 5),
        colgap = 25
    )

    metric_elements = [
        PolyElement(
            color = "#2171b5",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#238b45",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#2171b5",
            strokecolor = :transparent
        ),
        PolyElement(
            color = "#cb181d",
            strokecolor = :transparent
        ),
        LineElement(
            color = :gray25,
            linewidth = 2.5
        )
    ]

    metric_labels = [
        "LS calls",
        "LS improvements",
        "ΣΔQ",
        "ΣΔΦ",
        "LS execution time"
    ]

    Label(
        fig[6, 1:2],
        "METRIC REPRESENTATION",
        fontsize = 12,
        font = :bold,
        color = :gray25,
        halign = :center
    )

    Legend(
        fig[7, 1:2],
        metric_elements,
        metric_labels;
        orientation = :horizontal,
        framevisible = false,
        tellwidth = false,
        tellheight = true,
        halign = :center,
        labelsize = 11,
        patchsize = (22, 14),
        padding = (10, 5, 10, 5),
        colgap = 25
    )

    colgap!(fig.layout, 55)
    rowgap!(fig.layout, 15)

    out = joinpath(figure_root, "local_search")
    mkpath(out)

    save_figure(
        fig,
        joinpath(out, "ls_behavior_comparison.png")
    )

    return true
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

    try
    if fig_ls_behavior(df, figure_dir)
        made += 1
    end
    catch err
        @warn "Could not create Local Search behavior figure" exception=(
            err, catch_backtrace()
        )
    end

    try
    if fig_runtime_comparison(df, figure_dir)
        made += 1
    end
    catch err
        @warn "Could not create Runtime comparison figure" exception=(
            err, catch_backtrace()
        )
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