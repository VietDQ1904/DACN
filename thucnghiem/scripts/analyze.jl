using Pkg; Pkg.activate(joinpath(@__DIR__, "..")); using MOHGALS, CSV, DataFrames, Statistics, Printf, JSON3

function read_runs(paths)
    rows=DataFrame()
    raw=Vector{Dict{Symbol,Any}}()
    for path in paths
        for line in eachline(path)
            isempty(strip(line)) && continue
            x=JSON3.read(line)
            push!(raw,Dict(Symbol(k)=>v for (k,v) in pairs(x)))
        end
    end
    DataFrame(raw)
end
isnum(x)=x isa Real && !isnan(Float64(x))   # JSON null (nothing) / missing / NaN are skipped
mean_or_nan(x)=isempty(skipmissing(x)) ? NaN : mean(skipmissing(x))
std_or_nan(x)=length(collect(skipmissing(x)))<2 ? NaN : std(collect(skipmissing(x)))

out="results/report"; oi=findfirst(==("--out"),ARGS)
if oi!==nothing && oi<length(ARGS); out=ARGS[oi+1]; end
paths=String[]; skip=false
for (i,x) in enumerate(ARGS)
    if skip; skip=false; continue; end
    if x=="--out"; skip=true; continue; end
    push!(paths,x)
end
isempty(paths) && error("Usage: julia --project=. scripts/analyze.jl results/main/runs.jsonl ... --out results/report")
df=read_runs(paths); mkpath(out); mkpath(joinpath(out,"tables"))

metrics=[:q,:phi,:nmi,:ari,:runtime,:generations,:n_eval,:archive_size,:hv_final,:ls_calls,:ls_changed,:ls_improved,:ls_time,:ls_dq,:ls_dphi]
for m in metrics
    outdf=DataFrame()
    for sub in groupby(df,[:dataset,:algo])
        vals=Float64[Float64(x) for x in sub[!,m] if isnum(x)]
        row=DataFrame(dataset=[String(sub.dataset[1])],algo=[String(sub.algo[1])],n=[length(vals)],mean=[isempty(vals) ? NaN : mean(vals)],std=[length(vals)<2 ? NaN : std(vals)],median=[isempty(vals) ? NaN : median(vals)],q1=[isempty(vals) ? NaN : quantile(vals,0.25)],q3=[isempty(vals) ? NaN : quantile(vals,0.75)])
        append!(outdf,row)
    end
    CSV.write(joinpath(out,"tables",string(m,"_summary.csv")),outdf)
end

# Report-oriented tables mirroring the old Python analyzer.
for (name,algs) in [("t3_2_1_GA",["GA"]),("t3_2_2_HybridGA",["HGA"]),("t3_2_3_MO",["NSGA2","MO-HGA-LS"]),("t3_2_4_Baselines",["Louvain","Leiden"]),("t3_3_1_modularity",["GA","HGA","NSGA2","MO-HGA-LS","Louvain","Leiden"]),("t3_3_2_external",["GA","HGA","NSGA2","MO-HGA-LS","Louvain","Leiden"]),("t3_3_3_runtime",["GA","HGA","NSGA2","MO-HGA-LS","Louvain","Leiden"]),("t3_3_4_hv",["NSGA2","MO-HGA-LS"]),("t3_4_1_ls_effect",["GA","HGA","NSGA2","MO-HGA-LS"])]
    sel=filter(r->String(r.algo) in algs,eachrow(df)) |> collect
    isempty(sel) && continue
    od=DataFrame(sel)
    CSV.write(joinpath(out,"tables",name*".csv"),od)
end

# Dataset metadata from actual post-preprocessing graphs. This is deliberately generated from the same loader used by experiments.
data_dir=length(ARGS)>=2 ? "data" : "data"
datasets=unique(String.(df.dataset)); meta=DataFrame(dataset=String[],n=Int[],m=Int[],avg_degree=Float64[],avg_cc=Float64[],n_truth_labeled=Int[],n_truth_communities=Int[])
for d in datasets
    try
        g=load_dataset(d,data_dir); push!(meta,(d,g.n,g.m,g.stats["avg_degree"],g.stats["avg_cc"],get(g.stats,"n_truth_labeled",0),get(g.stats,"n_truth_communities",0)))
    catch e
        @warn "Could not reload $d for metadata" exception=e
    end
end
CSV.write(joinpath(out,"tables","dataset_metadata_postprocessed.csv"),meta)

# Compact markdown summary.
open(joinpath(out,"tables.md"),"w") do io
    println(io,"# Experimental report")
    println(io,"Generated from homogeneous runs only. HV is reported for NSGA2 and MO-HGA-LS; GA/HGA are compared by Q/NMI/ARI/runtime.")
    for d in sort(unique(String.(df.dataset)))
        println(io,"\n## $d\n")
        sub=df[df.dataset .== d,:]
        println(io,"| Algorithm | Q | Phi | NMI | ARI | Runtime (s) | HV |")
        println(io,"|---|---:|---:|---:|---:|---:|---:|")
        for a in unique(String.(sub.algo))
            x=sub[sub.algo .== a,:]
            fmt(metric)=begin; v=Float64[Float64(z) for z in x[!,metric] if isnum(z)]; isempty(v) ? "-" : @sprintf("%.4f ± %.4f",mean(v),length(v)>1 ? std(v) : 0.0) end
            println(io,"| $a | $(fmt(:q)) | $(fmt(:phi)) | $(fmt(:nmi)) | $(fmt(:ari)) | $(fmt(:runtime)) | $(a in ("NSGA2","MO-HGA-LS") ? fmt(:hv_final) : "-") |")
        end
    end
end
println("Wrote report tables to $out")
