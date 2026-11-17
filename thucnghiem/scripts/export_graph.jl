# Export graph structures to CSV.
# Run from the project root:
#   julia --project=. scripts/export_graphs.jl
# Optional dataset names:
#   julia --project=. scripts/export_graphs.jl karate polbooks football email

using MOHGALS
using CSV
using DataFrames
using Statistics

const DEFAULT_DATASETS = ["karate", "polbooks", "football", "email", "amazon", "polblogs"]

function graph_id(g)
    # Stable identifier for the loaded graph representation.
    return "$(g.name)_n$(g.n)_m$(g.m)"
end

function export_graph(g, outdir)
    gid = graph_id(g)

    nodes = DataFrame(
        graph_id = fill(gid, g.n),
        node_id = collect(1:g.n),
        degree = g.deg,
        clustering_coefficient = g.cc,
        ground_truth_community = g.truth === nothing ? fill(missing, g.n) : Union{Missing,Int}[x < 0 ? missing : x for x in g.truth]
    )

    edges = DataFrame(
        graph_id = fill(gid, g.m),
        source = g.edges_u,
        target = g.edges_v
    )

    CSV.write(joinpath(outdir, "$(g.name)_nodes.csv"), nodes)
    CSV.write(joinpath(outdir, "$(g.name)_edges.csv"), edges)

    return (
        graph_id = gid,
        dataset = g.name,
        n = g.n,
        m = g.m,
        average_degree = 2g.m / max(g.n, 1),
        average_clustering_coefficient = isempty(g.cc) ? missing : mean(g.cc),
        has_ground_truth = g.truth !== nothing
    )
end

function main()
    datasets = isempty(ARGS) ? DEFAULT_DATASETS : ARGS
    outdir = joinpath("results", "graphs")
    mkpath(outdir)
    metadata = NamedTuple[]

    for name in datasets
        println("Loading graph: ", name)
        try
            g = load_dataset(name)
            push!(metadata, export_graph(g, outdir))
            println("  Exported ", g.name, " (n=", g.n, ", m=", g.m, ")")
        catch err
            @warn "Could not export dataset" dataset=name exception=(err, catch_backtrace())
        end
    end

    isempty(metadata) && error("No graphs were exported. Check dataset files and data directory.")
    CSV.write(joinpath(outdir, "metadata.csv"), DataFrame(metadata))
    println("\nCSV graph files saved in: ", abspath(outdir))
end

main()
