function to_simplegraph(g::GraphData)
    sg=SimpleGraph(g.n); for (a,b) in zip(g.edges_u,g.edges_v); add_edge!(sg,a,b); end; sg
end

# Pure-Julia Louvain (modularity), randomised node order for seed-dependence.
function louvain_labels(g::GraphData,seed::Int)
    rng=MersenneTwister(seed); n=g.n; m2=2.0*g.m
    # level-0 weighted graph as adjacency lists with weights
    nbr=[copy(g.adj[u]) for u in 1:n]; wt=[ones(Float64,length(g.adj[u])) for u in 1:n]; selfw=zeros(n)
    node2comm=collect(1:n)           # original node -> current super-node
    while true
        N=length(nbr); comm=collect(1:N); k=[sum(wt[u])+2selfw[u] for u in 1:N]; tot=copy(k)
        improved=false; moved=true; order=collect(1:N)
        while moved
            moved=false; shuffle!(rng,order)
            for u in order
                cu=comm[u]; w=Dict{Int,Float64}()
                for (v,x) in zip(nbr[u],wt[u]); v==u && continue; w[comm[v]]=get(w,comm[v],0.0)+x; end
                tot[cu]-=k[u]; best=cu; bg=get(w,cu,0.0)-tot[cu]*k[u]/m2
                for (c,x) in w
                    gain=x-tot[c]*k[u]/m2
                    gain>bg+1e-12 && (bg=gain; best=c)
                end
                tot[best]+=k[u]
                if best!=cu; comm[u]=best; moved=true; improved=true; end
            end
        end
        improved || break
        cm=canonical(comm); K=maximum(cm)
        node2comm=[cm[c] for c in node2comm]
        nn=[Dict{Int,Float64}() for _ in 1:K]; ns=zeros(K)
        for u in 1:N
            cu=cm[u]; ns[cu]+=selfw[u]
            for (v,x) in zip(nbr[u],wt[u])
                cv=cm[v]
                if cu==cv; ns[cu]+=x/2 else nn[cu][cv]=get(nn[cu],cv,0.0)+x end
            end
        end
        nbr=[collect(keys(d)) for d in nn]; wt=[collect(values(d)) for d in nn]; selfw=ns
        K==N && break
    end
    canonical(node2comm)
end

function run_louvain(g::GraphData,seed)
    lab=louvain_labels(g,seed); q,p=q_phi(g,lab); return lab,q,p
end

function run_leiden(g::GraphData,seed)
    # Exact Leiden is intentionally kept as an optional backend. The core proposed method remains pure Julia.
    try
        @eval using PythonCall
    catch
        error("Leiden backend unavailable. Add PythonCall and install python-igraph/leidenalg, or run with --skip-leiden.")
    end
    ig=pyimport("igraph"); la=pyimport("leidenalg")
    edges=Py(collect(zip(g.edges_u .- 1,g.edges_v .- 1)))
    pg=ig.Graph(n=g.n, edges=edges, directed=false)
    part=la.find_partition(pg,la.ModularityVertexPartition,seed=seed)
    lab=Int.(pyconvert(Vector,part.membership)).+1; lab=canonical(lab); q,p=q_phi(g,lab); return lab,q,p
end

function run_baseline(g,algo,seed)
    t=time(); lab,q,p=algo=="Louvain" ? run_louvain(g,seed) : run_leiden(g,seed)
    return (lab=lab,q=q,phi=p,runtime=time()-t)
end
