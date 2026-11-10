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

    sortcols = [:dataset, :algo]
    :seed in propertynames(df) && push!(sortcols, :seed)
    sort!(df, sortcols)
    return df
end

function finite_values(df::DataFrame, algo::AbstractString, metric::Symbol)
    metric in propertynames(df) || return Float64[]
    result = Float64[]
    for value in df[df.algo .== algo, metric]
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

metric_values(df, algo, metric) = finite_values(df, string(algo), Symbol(metric))
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

function set_algorithm_ticks!(ax, algos)
    ax.xticks = (1:length(algos), [algo_label(a) for a in algos])
    ax.xticklabelrotation = pi / 7
    return nothing
end

function distribution_panel!(ax, sub, algos, metric, title_text; logscale=false, reference=nothing)
    any_data = false
    for (i, algo) in enumerate(algos)
        values = metric_values(sub, algo, metric)
        if logscale
            values = filter(>(0), values)
        end
        isempty(values) && continue
        any_data = true
        n = length(values)
        offsets = [i + 0.055 * sin(2pi * j / max(n, 1)) for j in 1:n]
        scatter!(ax, offsets, values;
            color=algo_color(algo), markersize=8,
            label=algo_label(algo),
        )
        m = mean(values)
        scatter!(ax, [i], [m];
            color=algo_color(algo), marker=:diamond,
            markersize=15, strokecolor=:black, strokewidth=1.2,
        )
    end
    if !any_data
        add_no_data!(ax, title_text)
        return
    end
    reference !== nothing && isfinite(reference) &&
        hlines!(ax, [reference]; color=:black, linestyle=:dash, linewidth=1.5)
    ax.title = title_text
    set_algorithm_ticks!(ax, algos)
    logscale && (ax.yscale = log10)
    return nothing
end

function mean_bar_panel!(ax, sub, algos, metric, title_text)
    means = [mean_or_nan(metric_values(sub, a, metric)) for a in algos]
    sds = [std_or_zero(metric_values(sub, a, metric)) for a in algos]
    valid = [i for i in eachindex(algos) if isfinite(means[i])]
    isempty(valid) && return add_no_data!(ax, title_text)

    for i in valid
        barplot!(ax, [i], [means[i]];
            color=algo_color(algos[i]), width=0.65,
        )
        errorbars!(ax, [i], [means[i]], [sds[i]];
            whiskerwidth=8, color=:black,
        )
        text!(ax, i, means[i] + sds[i];
            text=@sprintf("%.3f", means[i]),
            align=(:center, :bottom),
            fontsize=10,
        )
    end
    ax.title = title_text
    set_algorithm_ticks!(ax, algos)
    return nothing
end

function fig_dashboard(sub, dataset, out)
    algos = algorithms_in(sub)
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
        selected = metric == :n_eval ? filter(a -> a in GA_FAMILY, algos) : algos
        distribution_panel!(ax, sub, selected, metric, title_text; logscale=logscale)
    end
    Label(fig[0, :], "$dataset — $(nrow(sub) ÷ 6) experimental runs"; fontsize=21)
    save_figure(fig, joinpath(out, "dashboard.png"))
end

function fig_metric_bars(sub, dataset, out)
    algos = algorithms_in(sub)
    fig = Figure(size=(1450, 470))
    metrics = [(:q, "Q"), (:phi, "Φ"), (:nmi, "NMI"), (:ari, "ARI")]
    for (i, (metric, title_text)) in enumerate(metrics)
        ax = Axis(fig[1, i])
        mean_bar_panel!(ax, sub, algos, metric, "$title_text — mean ± SD")
    end
    Label(fig[0, :], "$dataset — quality metrics"; fontsize=20)
    save_figure(fig, joinpath(out, "bars_metrics.png"))
end

function fig_pareto(sub, dataset, out)
    (:q in propertynames(sub) && :phi in propertynames(sub)) || return
    # Mở rộng khung đồ thị bên phải một chút để đặt Legend hoàn toàn ra ngoài vùng vẽ dữ liệu
    fig = Figure(size=(950, 620))
    ax = Axis(fig[1, 1];
        title="$dataset — Modularity Q vs Conductance Φ",
        xlabel="Modularity Q (↑)", ylabel="Conductance Φ (↓)",
    )
    for algo in algorithms_in(sub)
        q = Float64[]
        p = Float64[]
        rows = sub[sub.algo .== algo, :]
        for row in eachrow(rows)
            qv, pv = row.q, row.phi
            if !ismissing(qv) && !ismissing(pv) && isfinite(qv) && isfinite(pv)
                push!(q, qv)
                push!(p, pv)
            end
        end
        isempty(q) && continue
        scatter!(ax, q, p; color=algo_color(algo), label=algo_label(algo), markersize=10)
    end
    # Đặt chú thích nằm ở cột bên phải ngoài trục tọa độ chính
    Legend(fig[1, 2], ax; framevisible=true)
    save_figure(fig, joinpath(out, "pareto.png"))
end


function fig_time_and_ls(sub, dataset, out)
    algos = algorithms_in(sub)
    fig = Figure(size=(1250, 480))

    ax1 = Axis(
        fig[1, 1];
        title="Execution time (log scale)",
        xlabel="Algorithm",
        ylabel="Seconds",
        yscale=log10,
    )

    any_runtime = false
    positive_means = Float64[]

    for (i, algo) in enumerate(algos)
        values = filter(>(0), metric_values(sub, algo, :runtime))
        isempty(values) && continue

        m = mean(values)
        sd = std_or_zero(values)

        # Logarithmic axis requires strictly positive values.
        m > 0 || continue

        any_runtime = true
        push!(positive_means, m)

        barplot!(
            ax1, [i], [m];
            color=algo_color(algo),
            width=0.65,
        )

        lower = max(m - sd, m * 0.01)
        upper = max(m + sd, m * 1.01)

        errorbars!(
            ax1, [i], [m], [m - lower], [upper - m];
            whiskerwidth=8,
            color=:black,
        )
    end

    if any_runtime
        set_algorithm_ticks!(ax1, algos)

        # Ensure the axis always has a valid positive range.
        ymin = max(minimum(positive_means) * 0.5, eps(Float64))
        ymax = maximum(positive_means) * 2.0

        ylims!(ax1, ymin, ymax)
    else
        add_no_data!(ax1, "Execution time")
    end

    ax2 = Axis(
        fig[1, 2];
        title="Local-search time breakdown",
        xlabel="Algorithm",
        ylabel="Mean time (s)",
    )

    family = [a for a in algos if a in GA_FAMILY]
    any_ls = false

    for (i, algo) in enumerate(family)
        total_values = metric_values(sub, algo, :runtime)
        isempty(total_values) && continue

        ls_values = metric_values(sub, algo, :ls_time)

        total = mean(total_values)
        ls = isempty(ls_values) ? 0.0 : mean(ls_values)

        # Keep the breakdown consistent with total runtime.
        total = max(total, 0.0)
        ls = clamp(ls, 0.0, total)

        rest = max(total - ls, 0.0)

        barplot!(
            ax2, [i - 0.18], [rest];
            color="#56b4e9",
            width=0.34,
        )

        barplot!(
            ax2, [i + 0.18], [ls];
            color="#d55e00",
            width=0.34,
        )

        ratio = total > 0 ? 100 * ls / total : 0.0

        text!(
            ax2, i, total;
            text=@sprintf("%.1f%% LS", ratio),
            align=(:center, :bottom),
            fontsize=10,
        )

        any_ls = true
    end

    if any_ls
        set_algorithm_ticks!(ax2, family)

        # Legend for stacked components.
        elements = [
            PolyElement(color="#56b4e9"),
            PolyElement(color="#d55e00"),
        ]

        Legend(
            fig[2, 2], elements, ["Other execution time", "Local-search time"];
            orientation=:horizontal,
            framevisible=false,
        )
    else
        add_no_data!(ax2, "Local-search time")
    end

    save_figure(fig, joinpath(out, "time_cost.png"))
end


function fig_quality_vs_time(sub, dataset, out)
    algos = algorithms_in(sub)
    # Mở rộng chiều rộng để bảng Legend nằm sát bên phải ngoài biểu đồ thay vì đè lên đường đi
    fig = Figure(size=(1350, 430))
    for (i, (metric, title_text)) in enumerate([
        (:q, "Q vs runtime"), (:nmi, "NMI vs runtime"), (:hv_final, "Hypervolume vs runtime")
    ])
        ax = Axis(fig[1, i]; title="$title_text — $dataset",
            xlabel="Runtime (s)", ylabel=string(metric), xscale=log10)
        for algo in algos
            xs, ys = Float64[], Float64[]
            for row in eachrow(sub[sub.algo .== algo, :])
                x, y = row.runtime, row[metric]
                if !ismissing(x) && !ismissing(y) && isfinite(x) && x > 0 && isfinite(y)
                    push!(xs, x)
                    push!(ys, y)
                end
            end
            isempty(xs) && continue
            scatter!(ax, xs, ys; color=algo_color(algo), label=algo_label(algo), markersize=9)
        end
        # Đặt chú thích nằm ở góc phải ngoài cùng hoặc dùng khoảng trống phù hợp
        axislegend(ax; position=:lt, backgroundcolor=:white)
    end
    save_figure(fig, joinpath(out, "quality_vs_time.png"))
end

function fig_q_vs_nmi(sub, dataset, out)
    (:q in propertynames(sub) && :nmi in propertynames(sub)) || return
    fig = Figure(size=(820, 540))
    ax = Axis(fig[1, 1]; title="$dataset — Q vs NMI", xlabel="Modularity Q", ylabel="NMI")
    for algo in algorithms_in(sub)
        xs, ys = Float64[], Float64[]
        for row in eachrow(sub[sub.algo .== algo, :])
            x, y = row.q, row.nmi
            if !ismissing(x) && !ismissing(y) && isfinite(x) && isfinite(y)
                push!(xs, x)
                push!(ys, y)
            end
        end
        isempty(xs) || scatter!(ax, xs, ys; color=algo_color(algo), label=algo_label(algo))
    end
    # Đưa chú thích ra phía ngoài bên phải của biểu đồ
    Legend(fig[1, 2], ax; framevisible=true)
    save_figure(fig, joinpath(out, "q_vs_nmi.png"))
end

function fig_ls_comparison(sub, dataset, out)
    comparisons = [("GA", "HGA", "Single-objective GA vs hybrid"),
                   ("NSGA2", "MO-HGA-LS", "Multi-objective NSGA-II vs hybrid")]
    comparisons = [p for p in comparisons
        if !isempty(metric_values(sub, p[1], :q)) && !isempty(metric_values(sub, p[2], :q))]
    isempty(comparisons) && return

    metrics = [(:q, "Q"), (:hv_final, "Hypervolume"), (:nmi, "NMI"), (:runtime, "Runtime (s)")]
    fig = Figure(size=(1300, 440))
    for (j, (metric, title_text)) in enumerate(metrics)
        ax = Axis(fig[1, j]; title=title_text)
        for (i, (a, b, _)) in enumerate(comparisons)
            va = metric_values(sub, a, metric)
            vb = metric_values(sub, b, metric)
            isempty(va) && continue
            isempty(vb) && continue
            x1, x2 = i - 0.18, i + 0.18
            barplot!(ax, [x1], [mean(va)]; color=algo_color(a), width=0.34)
            barplot!(ax, [x2], [mean(vb)]; color=algo_color(b), width=0.34)
        end
        ax.xticks = (1:length(comparisons), [p[3] for p in comparisons])
        ax.xticklabelrotation = pi / 10
    end
    save_figure(fig, joinpath(out, "ls_ablation.png"))
end

function fig_relative_scores(sub, dataset, out)
    algos = algorithms_in(sub)
    specs = [
        (:q, true, "Q"), (:phi, false, "Φ"), (:nmi, true, "NMI"),
        (:ari, true, "ARI"), (:hv_final, true, "HV"), (:runtime, false, "Runtime"),
    ]
    names = String[]
    rows = Vector{Float64}[]
    for (metric, higher_is_better, title_text) in specs
        means = [mean_or_nan(metric_values(sub, a, metric)) for a in algos]
        finite = filter(isfinite, means)
        isempty(finite) && continue
        lo, hi = minimum(finite), maximum(finite)
        scores = [!isfinite(x) ? NaN : (hi == lo ? 1.0 : (x - lo) / (hi - lo)) for x in means]
        !higher_is_better && (scores = [isfinite(x) ? 1.0 - x : NaN for x in scores])
        push!(rows, scores)
        push!(names, title_text)
    end
    isempty(rows) && return

    matrix = reduce(vcat, (permutedims(r) for r in rows))
    fig = Figure(size=(max(800, 135 * length(algos)), max(360, 75 * length(names))))
    ax = Axis(fig[1, 1];
        title="$dataset — normalized scores (1 = best)",
        xticks=(1:length(algos), [algo_label(a) for a in algos]),
        yticks=(1:length(names), names),
    )
    hm = heatmap!(ax, 1:length(algos), 1:length(names), matrix;
        colormap=:YlGn, colorrange=(0, 1),
    )
    for i in axes(matrix, 1), j in axes(matrix, 2)
        isfinite(matrix[i, j]) || continue
        text!(ax, j, i; text=@sprintf("%.2f", matrix[i, j]),
            align=(:center, :center), fontsize=11)
    end
    Colorbar(fig[1, 2], hm)
    save_figure(fig, joinpath(out, "relative_scores.png"))
end

function per_dataset_figures(df, dataset, figure_root)
    sub = df[df.dataset .== dataset, :]
    out = joinpath(figure_root, safe_name(dataset))
    mkpath(out)
    jobs = [
        ("dashboard", () -> fig_dashboard(sub, dataset, out)),
        ("bars", () -> fig_metric_bars(sub, dataset, out)),
        ("pareto", () -> fig_pareto(sub, dataset, out)),
        ("time_cost", () -> fig_time_and_ls(sub, dataset, out)),
        ("quality_vs_time", () -> fig_quality_vs_time(sub, dataset, out)),
        ("q_vs_nmi", () -> fig_q_vs_nmi(sub, dataset, out)),
        ("ls_ablation", () -> fig_ls_comparison(sub, dataset, out)),
        ("relative_scores", () -> fig_relative_scores(sub, dataset, out)),
    ]
    made = String[]
    for (name, job) in jobs
        try
            job()
            push!(made, name)
        catch err
            @warn "Could not create figure" dataset figure=name exception=(err, catch_backtrace())
        end
    end
    return made
end

function summary_figures(df, figure_root)
    out = joinpath(figure_root, "summary")
    mkpath(out)
    datasets = sort(unique(df.dataset))
    algos = algorithms_in(df)

    # 1. Biểu đồ tổng hợp các chỉ số chính (Q, NMI, ARI, Runtime) theo Dataset
    fig = Figure(size=(1600, 900))
    g_plots = fig[1:2, 1:2] = GridLayout()
    
    specs = [(:q, "Modularity Q"), (:nmi, "NMI"), (:ari, "ARI"), (:runtime, "Runtime (s)")]
    
    for (i, (metric, title_text)) in enumerate(specs)
        r = (i - 1) ÷ 2 + 1
        c = (i - 1) % 2 + 1
        ax = Axis(g_plots[r, c]; title=title_text)
        plotted = false
        for (j, algo) in enumerate(algos)
            for (k, dataset) in enumerate(datasets)
                values = metric_values(df[df.dataset .== dataset, :], algo, metric)
                if metric == :runtime
                    values = filter(>(0), values)
                end
                isempty(values) && continue
                x = k + (j - (length(algos) + 1) / 2) * 0.78 / max(length(algos), 1)
                
                m_val = mean(values)
                sd_val = std_or_zero(values)
                
                barplot!(ax, [x], [m_val];
                    width=0.78 / max(length(algos), 1),
                    color=algo_color(algo),
                )
                
                # An toàn hóa error bar cho trục logarit (tránh giá trị <= 0)
                if metric == :runtime
                    low_err = min(sd_val, m_val - 1e-4)
                    low_err = max(low_err, 1e-5)
                    errorbars!(ax, [x], [m_val], [low_err], [sd_val]; color=:black)
                else
                    errorbars!(ax, [x], [m_val], [sd_val]; color=:black)
                end
                
                plotted = true
            end
        end
        plotted || add_no_data!(ax, title_text)
        ax.xticks = (1:length(datasets), datasets)
        metric == :runtime && (ax.yscale = log10)
    end

    # Tạo Legend độc lập dựa trên danh sách thuật toán hiện có (tránh hoàn toàn lỗi trùng lặp vàUndefVarError)
    legend_elements = [PolyElement(color=algo_color(a), strokecolor=:black, strokewidth=1) for a in algos]
    legend_labels = [algo_label(a) for a in algos]
    Legend(fig[1:2, 3], legend_elements, legend_labels; title="Algorithms", framevisible=true)

    Label(fig[0, 1:2], "Cross-dataset comparison — mean ± SD"; fontsize=20, font=:bold)
    save_figure(fig, joinpath(out, "all_datasets_metrics.png"))

    # 2. Biểu đồ tổng hợp Hypervolume (hv_final) across datasets
    if :hv_final in propertynames(df)
        fig_hv = Figure(size=(1200, 520))
        ax_hv = Axis(fig_hv[1, 1]; title="Cross-dataset Hypervolume (hv_final) comparison", ylabel="Hypervolume")
        plotted_hv = false
        for (j, algo) in enumerate(algos)
            for (k, dataset) in enumerate(datasets)
                values = metric_values(df[df.dataset .== dataset, :], algo, :hv_final)
                isempty(values) && continue
                x = k + (j - (length(algos) + 1) / 2) * 0.78 / max(length(algos), 1)
                
                m_val = mean(values)
                sd_val = std_or_zero(values)
                
                barplot!(ax_hv, [x], [m_val];
                    width=0.78 / max(length(algos), 1),
                    color=algo_color(algo),
                )
                errorbars!(ax_hv, [x], [m_val], [sd_val]; color=:black)
                plotted_hv = true
            end
        end
        plotted_hv || add_no_data!(ax_hv, "Hypervolume")
        ax_hv.xticks = (1:length(datasets), datasets)
        
        Legend(fig_hv[1, 2], legend_elements, legend_labels; title="Algorithms", framevisible=true)
        Label(fig_hv[0, 1], "Cross-dataset Hypervolume comparison — mean ± SD"; fontsize=20, font=:bold)
        save_figure(fig_hv, joinpath(out, "all_datasets_hypervolume.png"))
    end

    # 3. Mean ranks cho Q và NMI
    fig_rank = Figure(size=(1050, 470))
    for (j, metric) in enumerate((:q, :nmi))
        ranks = Dict(algo => Float64[] for algo in algos)
        for dataset in datasets
            means = Dict(algo => mean_or_nan(metric_values(df[df.dataset .== dataset, :], algo, metric))
                         for algo in algos)
            valid = [algo for algo in algos if isfinite(means[algo])]
            length(valid) < 2 && continue
            sort!(valid; by=algo -> means[algo], rev=true)
            for (rank, algo) in enumerate(valid)
                push!(ranks[algo], Float64(rank))
            end
        end
        ax = Axis(fig_rank[1, j]; title="Mean rank — $(uppercase(string(metric)))",
            ylabel="Rank (1 = best)")
        plotted = false
        for (i, algo) in enumerate(algos)
            isempty(ranks[algo]) && continue
            value = mean(ranks[algo])
            barplot!(ax, [i], [value]; color=algo_color(algo), width=0.65)
            text!(ax, i, value; text=@sprintf("%.2f", value),
                align=(:center, :bottom), fontsize=10)
            plotted = true
        end
        plotted || add_no_data!(ax, "Mean rank")
        set_algorithm_ticks!(ax, algos)
    end
    save_figure(fig_rank, joinpath(out, "mean_rank.png"))

    # 4. Runtime vs graph size
    if :m in propertynames(df)
        fig_size = Figure(size=(950, 560))
        ax = Axis(fig_size[1, 1]; title="Runtime vs graph size",
            xlabel="Number of edges m", ylabel="Mean runtime (s)",
            xscale=log10, yscale=log10)
        for algo in algos
            xs, ys = Float64[], Float64[]
            for dataset in datasets
                sub = df[df.dataset .== dataset, :]
                values = filter(>(0), metric_values(sub, algo, :runtime))
                isempty(values) && continue
                edge_values = Float64[]
                for x in skipmissing(sub.m)
                    y = tryparse(Float64, string(x))
                    y !== nothing && isfinite(y) && y > 0 && push!(edge_values, y)
                end
                isempty(edge_values) && continue
                push!(xs, first(edge_values))
                push!(ys, mean(values))
            end
            isempty(xs) && continue
            order = sortperm(xs)
            lines!(ax, xs[order], ys[order]; color=algo_color(algo))
            scatter!(ax, xs[order], ys[order]; color=algo_color(algo))
        end
        
        line_elements = [LineElement(color=algo_color(a), linewidth=2) for a in algos]
        Legend(fig_size[1, 2], line_elements, legend_labels; title="Algorithms", framevisible=true)
        save_figure(fig_size, joinpath(out, "time_vs_size.png"))
    end
end


function write_tables(df, table_root)
    mkpath(table_root)
    CSV.write(joinpath(table_root, "runs_normalized.csv"), df)

    summary = DataFrame(
        dataset=String[], algo=String[], runs=Int[],
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

    for dataset in sort(unique(df.dataset)), algo in algorithms_in(df)
        sub = df[(df.dataset .== dataset) .& (df.algo .== algo), :]
        nrow(sub) == 0 && continue
        row = Dict{Symbol,Any}(:dataset => dataset, :algo => algo, :runs => nrow(sub))
        for metric in keys(METRIC_ALIASES)
            metric in propertynames(df) || continue
            values = metric_values(sub, algo, metric)
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
    println("Datasets  : ", join(sort(unique(df.dataset)), ", "))
    println("Algorithms: ", join(algorithms_in(df), ", "))

    write_tables(df, table_dir)

    made = 0
    for dataset in sort(unique(df.dataset))
        made += length(per_dataset_figures(df, dataset, figure_dir))
    end

    try
        summary_figures(df, figure_dir)
        made += 3 + (:m in propertynames(df) ? 1 : 0)
    catch err
        @warn "Could not create cross-dataset summary figures" exception=(err, catch_backtrace())
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