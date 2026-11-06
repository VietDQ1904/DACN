using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON3
using CSV
using DataFrames
using Statistics
using CairoMakie


input_file = length(ARGS) >= 1 ? ARGS[1] : "results/main/runs.jsonl"
output_dir = length(ARGS) >= 2 ? ARGS[2] : "results/report"

fig_dir = joinpath(output_dir, "figures")
table_dir = joinpath(output_dir, "tables")

mkpath(fig_dir)
mkpath(table_dir)

function object_to_dict(obj)
return Dict{String,Any}(String(k) => v for (k, v) in pairs(obj))
end

function load_runs(path::AbstractString)
isfile(path) || error("Input file does not exist: $path")
raw = strip(read(path, String))
isempty(raw) && error("Input file is empty: $path")

cleaned = replace(raw, r",\s*\]" => "]")

try
    parsed = JSON3.read(cleaned)
    parsed isa AbstractVector || error("Expected a JSON array.")
    return [object_to_dict(x) for x in parsed]
catch array_error
    rows = Dict{String,Any}[]
    for line in eachline(IOBuffer(raw))
        s = strip(line)
        isempty(s) && continue
        s in ("[", "]", ",") && continue
        startswith(s, "[") && (s = strip(s[2:end]))
        endswith(s, ",") && (s = strip(s[1:end-1]))
        endswith(s, "]") && (s = strip(s[1:end-1]))
        isempty(s) && continue

        try
            obj = JSON3.read(s)
            push!(rows, object_to_dict(obj))
        catch line_error
            error("Could not parse line: $(sprint(showerror, line_error))")
        end
    end
    isempty(rows) && error("No valid records found.")
    return rows
end


end

rows = load_runs(input_file)
df = DataFrame(rows)

for col in names(df)
if any(x -> x === nothing || ismissing(x), df[!, col])
df[!, col] = map(x -> x === nothing ? missing : x, df[!, col])
end
end

println("Loaded $(nrow(df)) experiment records from:$input_file")

df.dataset = string.(df.dataset)
df.algo = string.(df.algo)

sort_cols = [:dataset, :algo]
"seed" in names(df) && push!(sort_cols, :seed)
sort!(df, sort_cols)

datasets = sort(unique(df.dataset))
algorithms = sort(unique(df.algo))

println("Datasets  : ", join(datasets, ", "))
println("Algorithms: ", join(algorithms, ", "))

report_metrics = [
:q, :phi, :nmi, :ari, :runtime,
:hv_final, :n_eval, :ls_time,
:archive_size, :ls_calls, :ls_changed, :ls_improved
]

for metric in report_metrics
if metric in propertynames(df)
df[!, metric] = map(df[!, metric]) do x
ismissing(x) || x === nothing ? missing : Float64(x)
end
end
end

CSV.write(joinpath(table_dir, "runs_normalized.csv"), df; transform = (c, v) -> v === nothing ? missing : v)

summary_rows = NamedTuple[]
for dataset in datasets
for algo in algorithms
sub = df[(df.dataset .== dataset) .& (df.algo .== algo), :]
isempty(sub) && continue

    result = Dict{Symbol,Any}(:dataset => dataset, :algo => algo, :runs => nrow(sub))
    for metric in report_metrics
        metric in propertynames(df) || continue
        vals = Float64[x for x in skipmissing(sub[!, metric]) if isfinite(x)]
        result[Symbol(metric, :_mean)] = isempty(vals) ? NaN : mean(vals)
        result[Symbol(metric, :_std)] = isempty(vals) ? NaN : (length(vals) > 1 ? std(vals) : 0.0)
    end
    push!(summary_rows, (; result...))
end


end

summary_df = DataFrame(summary_rows)
CSV.write(joinpath(table_dir, "runs_summary.csv"), summary_df; transform = (c, v) -> v === nothing ? missing : v)


function plot_global_grouped_bar_external_legend(summary_df, datasets, algorithms, metric, title_text, ylabel_text, filename)
mean_col = Symbol(metric, :_mean)
mean_col in propertynames(summary_df) || return false

fig = Figure(size = (1200, 650))
ax = Axis(fig[1, 1], title = title_text, xlabel = "Dataset", ylabel = ylabel_text,
          xticks = (1:length(datasets), datasets))

n_algos = length(algorithms)
group_width = 0.8
bar_width = group_width / n_algos

plots = []
for (i, algo) in enumerate(algorithms)
    means = Float64[]
    for d in datasets
        row = summary_df[(summary_df.dataset .== d) .& (summary_df.algo .== algo), :]
        push!(means, (nrow(row) > 0 && isfinite(row[1, mean_col])) ? row[1, mean_col] : 0.0)
    end
    x_pos = (1:length(datasets)) .+ (i - (n_algos + 1) / 2) * bar_width
    p = barplot!(ax, x_pos, means; width = bar_width)
    push!(plots, p)
end

Legend(fig[1, 2], plots, algorithms, "Algorithms"; framevisible = true)
save(joinpath(fig_dir, filename), fig)
return true


end

global figure_count = 0
println("\nGenerating optimized global summary charts...")


plot_global_grouped_bar_external_legend(summary_df, datasets, algorithms, :q,
"Tổng quan Modularity (Q) với các datasets và thuật toán", "Modularity Q", "global_01_modularity_q.png") && (global figure_count += 1)


plot_global_grouped_bar_external_legend(summary_df, datasets, algorithms, :phi,
"Tổng quan Conductance (Phi) với các datasets và thuật toán", "Conductance Phi", "global_02_conductance_phi.png") && (global figure_count += 1)

plot_global_grouped_bar_external_legend(summary_df, datasets, algorithms, :nmi,
"Tổng quan Normalized Mutual Information (NMI)", "NMI Score", "global_03_nmi.png") && (global figure_count += 1)


plot_global_grouped_bar_external_legend(summary_df, datasets, algorithms, :runtime,
"Tổng quan thời gian thực thi với các datasets và thuật toán", "Thời gian (giây)", "global_04_runtime.png") && (global figure_count += 1)


let
if :phi_mean in propertynames(summary_df) && :q_mean in propertynames(summary_df)
fig = Figure(size = (1200, 650))
ax = Axis(fig[1, 1], title = "So sánh cân bằng giữa Conductance (Phi) và Modularity (Q)",
xlabel = "Conductance Phi (thấp hơn là tốt hơn)", ylabel = "Modularity Q (cao hơn là tốt hơn)")

    plots = []
    for algo in algorithms
        sub_algo = summary_df[summary_df.algo .== algo, :]
        isempty(sub_algo) && continue
        x_vals = sub_algo.phi_mean
        y_vals = sub_algo.q_mean
        
        valid_idx = .!(isnan.(x_vals) .|| isnan.(y_vals))
        if any(valid_idx)
            p = scatter!(ax, x_vals[valid_idx], y_vals[valid_idx]; markersize = 16)
            push!(plots, p)
        end
    end
    Legend(fig[1, 2], plots, algorithms, "Algorithms"; framevisible = true)
    save(joinpath(fig_dir, "global_05_scatter_phi_vs_q.png"), fig)
    global figure_count += 1
end


end


let
fig = Figure(size = (1200, 650))
ax = Axis(fig[1, 1], title = "So sánh hiệu năng: Modularity Q và Runtime",
xlabel = "Thời gian trung bình (s)", ylabel = "Modularity Q trung bình")

plots = []
for algo in algorithms
    sub_algo = summary_df[summary_df.algo .== algo, :]
    isempty(sub_algo) && continue
    x_vals = sub_algo.runtime_mean
    y_vals = sub_algo.q_mean
    
    valid_idx = .!(isnan.(x_vals) .|| isnan.(y_vals))
    if any(valid_idx)
        p = scatter!(ax, x_vals[valid_idx], y_vals[valid_idx]; markersize = 16)
        push!(plots, p)
    end
end
Legend(fig[1, 2], plots, algorithms, "Algorithms"; framevisible = true)
save(joinpath(fig_dir, "global_06_scatter_q_runtime.png"), fig)
global figure_count += 1


end


let
if :n_eval_mean in propertynames(summary_df) && :nmi_mean in propertynames(summary_df)
fig = Figure(size = (1200, 650))
ax = Axis(fig[1, 1], title = "So sánh hiệu suất tìm kiếm: NMI vs Số lần đánh giá",
xlabel = "Số lần đánh giá trung bình", ylabel = "Điểm NMI trung bình")

    plots = []
    for algo in algorithms
        sub_algo = summary_df[summary_df.algo .== algo, :]
        isempty(sub_algo) && continue
        x_vals = sub_algo.n_eval_mean
        y_vals = sub_algo.nmi_mean
        
        valid_idx = .!(isnan.(x_vals) .|| isnan.(y_vals))
        if any(valid_idx)
            p = scatter!(ax, x_vals[valid_idx], y_vals[valid_idx]; markersize = 16)
            push!(plots, p)
        end
    end
    Legend(fig[1, 2], plots, algorithms, "Algorithms"; framevisible = true)
    save(joinpath(fig_dir, "global_07_scatter_nmi_evals.png"), fig)
    global figure_count += 1
end


end


let
fig = Figure(size = (1000, 650))
ax = Axis(fig[1, 1], title = "Tỷ lệ bản ghi theo Dataset (Biểu đồ tròn)", aspect = DataAspect())
hidedecorations!(ax)
hidespines!(ax)

counts = [sum(df.dataset .== d) for d in datasets]
total_runs = sum(counts)
percentages = round.((counts ./ total_runs) .* 100, digits = 1)
labels_with_pct = ["$d ($(pct)%)" for (d, pct) in zip(datasets, percentages)]

palette = [:cornflowerblue, :orange, :seagreen, :indianred]
colors = palette[1:min(length(datasets), length(palette))]

pie!(ax, counts; color = colors)
Legend(fig[1, 2], [PolyElement(color = c) for c in colors], labels_with_pct, "Datasets"; framevisible = true)

save(joinpath(fig_dir, "global_08_dataset_pie_chart.png"), fig)
global figure_count += 1


end


let
fig = Figure(size = (1000, 650))
ax = Axis(fig[1, 1], title = "Tỷ lệ chạy thuật toán (Biểu đồ tròn)", aspect = DataAspect())
hidedecorations!(ax)
hidespines!(ax)

algo_counts = [sum(df.algo .== a) for a in algorithms]
total_runs = sum(algo_counts)
percentages = round.((algo_counts ./ total_runs) .* 100, digits = 1)
labels_with_pct = ["$a ($(pct)%)" for (a, pct) in zip(algorithms, percentages)]

palette = [:dodgerblue, :gold, :mediumseagreen, :orchid, :skyblue, :chocolate]
colors = palette[1:min(length(algorithms), length(palette))]

pie!(ax, algo_counts; color = colors)
Legend(fig[1, 2], [PolyElement(color = c) for c in colors], labels_with_pct, "Algorithms"; framevisible = true)

save(joinpath(fig_dir, "global_09_algorithm_pie_chart.png"), fig)
global figure_count += 1


end

println()
println("=" ^ 68)
println("REPORT GENERATION COMPLETED SUCCESSFULLY")
println("=" ^ 68)
println("Total Records : ", nrow(df))
println("Global Figures: ", figure_count)
println("Output Dir    : ", fig_dir)
println("=" ^ 68)