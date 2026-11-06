struct GraphData
    n::Int
    m::Int
    adj::Vector{Vector{Int}}
    edges_u::Vector{Int}
    edges_v::Vector{Int}
    deg::Vector{Int}
    cc::Vector{Float64}
    truth::Union{Nothing,Vector{Int}}
    name::String
    stats::Dict{String,Any}
end

function build_graph(n::Int, raw_edges::Vector{Tuple{Int,Int}}; truth=nothing, name="", compute_cc=true)
    ed = Set{Tuple{Int,Int}}()
    for (a,b) in raw_edges
        a == b && continue
        x,y = a < b ? (a,b) : (b,a)
        1 <= x <= n && 1 <= y <= n && push!(ed,(x,y))
    end
    es = sort!(collect(ed))
    adj = [Int[] for _ in 1:n]
    eu = Int[]; ev = Int[]
    for (a,b) in es
        push!(adj[a],b); push!(adj[b],a); push!(eu,a); push!(ev,b)
    end
    deg = length.(adj)
    cc = zeros(Float64,n)
    if compute_cc
        marks=zeros(Int,n); stamp=0
        for u in 1:n
            d=deg[u]
            d < 2 && continue
            stamp+=1
            for v in adj[u]; marks[v]=stamp; end
            links=0
            for i in 1:d-1
                vi=adj[u][i]
                for j in i+1:d
                    links += marks[adj[u][j]]==stamp && insorted(adj[u][j],adj[vi])
                end
            end
            cc[u]=2links/(d*(d-1))
        end
    end
    stats = Dict{String,Any}("n"=>n,"m"=>length(es),"avg_degree"=>2length(es)/max(n,1),"avg_cc"=>compute_cc ? mean(cc) : NaN)
    truth2 = truth === nothing ? nothing : copy(truth)
    if truth2 !== nothing
        valid = [x for x in truth2 if x >= 0]
        stats["n_truth_communities"] = isempty(valid) ? 0 : length(unique(valid))
        stats["n_truth_labeled"] = length(valid)
    end
    return GraphData(n,length(es),adj,eu,ev,deg,cc,truth2,String(name),stats)
end

function largest_component(n, edges)
    adj=[Int[] for _ in 1:n]
    for (a,b) in edges; push!(adj[a],b); push!(adj[b],a); end
    seen=falses(n); best=Int[]
    for s in 1:n
        seen[s] && continue
        q=[s]; seen[s]=true; comp=Int[]
        for u in q
            push!(comp,u)
            for v in adj[u]
                if !seen[v]; seen[v]=true; push!(q,v); end
            end
        end
        length(comp)>length(best) && (best=comp)
    end
    keep=Set(best); mp=Dict(old=>i for (i,old) in enumerate(best))
    e2=Tuple{Int,Int}[(mp[a],mp[b]) for (a,b) in edges if a in keep && b in keep]
    return length(best), e2, mp
end

function decode(g::GraphData, genes::Vector{Int})
    # Weakly connected components of the functional graph u -> genes[u] (union-find).
    p=collect(1:g.n)
    @inline function root(x)
        while p[x]!=x; p[x]=p[p[x]]; x=p[x]; end; x
    end
    @inbounds for u in 1:g.n
        a=root(u); b=root(genes[u]); a!=b && (p[a]=b)
    end
    mp=zeros(Int,g.n); lab=zeros(Int,g.n); k=0
    @inbounds for u in 1:g.n
        r=root(u)
        if mp[r]==0; k+=1; mp[r]=k; end
        lab[u]=mp[r]
    end
    return lab
end

function community_stats(g::GraphData, labels::Vector{Int})
    K=maximum(labels); l=zeros(Int,K); d=zeros(Int,K)
    # FIX: volume d[c] = sum of node degrees in c. The old code added deg[a],deg[b]
    # once per EDGE, inflating d by roughly the degree and giving Q << -1 and phi > 1.
    @inbounds for u in 1:g.n
        d[labels[u]]+=g.deg[u]
    end
    @inbounds for i in eachindex(g.edges_u)
        ca=labels[g.edges_u[i]]
        ca==labels[g.edges_v[i]] && (l[ca]+=1)
    end
    return l,d
end

function conductance(g::GraphData, l::Int, d::Int)
    l==0 && return 1.0
    cut=d-2l; den=min(d,2g.m-d)
    den<=0 && return 1.0
    return cut/den
end

function q_phi_from(g::GraphData, l::Vector{Int}, d::Vector{Int})
    K=length(l); m=float(g.m); m2=2m
    q=0.0; ph=0.0
    @inbounds for i in 1:K
        q+=l[i]/m-(d[i]/m2)^2; ph+=conductance(g,l[i],d[i])
    end
    phi=ph/K
    # A one-community partition has modularity exactly zero under the
    # standard Newman-Girvan definition.  Conductance is conventionally 1
    # when the denominator min(vol(C), 2m-vol(C)) is zero.
    if K<=1; q=0.0; phi=1.0; end
    return q,phi
end

function q_phi(g::GraphData, labels::Vector{Int})
    l,d=community_stats(g,labels)
    return q_phi_from(g,l,d)
end

function canonical(labels::Vector{Int})
    mp=Dict{Int,Int}(); out=similar(labels); k=0
    for i in eachindex(labels)
        c=labels[i]
        if !haskey(mp,c); k+=1; mp[c]=k; end
        out[i]=mp[c]
    end
    return out
end

conductance(g::GraphData, l_val::Real, d_val::Real) = conductance(g, Int(round(l_val)), Int(round(d_val)))
