struct LARFactory
    g::GraphData
    high::Vector{Vector{Int}}
    rho::Float64
end

function LARFactory(g::GraphData,rho)
    hi=Vector{Vector{Int}}(undef,g.n)
    for u in 1:g.n
        nbs=g.adj[u]
        μ=mean(g.cc[nbs])
        hi[u]=[v for v in nbs if g.cc[v] >= μ-1e-12]
        isempty(hi[u]) && (hi[u]=copy(nbs))
    end
    LARFactory(g,hi,rho)
end

function random_individual(f::LARFactory,rng)
    g=f.g; genes=Vector{Int}(undef,g.n)
    for u in 1:g.n
        pool = rand(rng)<f.rho ? f.high[u] : g.adj[u]
        genes[u]=pool[rand(rng,1:length(pool))]
    end
    genes
end
init_population(f,μ,rng)=[random_individual(f,rng) for _ in 1:μ]

# uniform crossover: one RNG call gives 64 coin flips (same distribution as rand()<0.5 per gene)
function crossover(a,b,rng)
    n=length(a); out=similar(a); i=1
    @inbounds while i<=n
        bits=rand(rng,UInt64); stop=min(n,i+63)
        for k in i:stop
            out[k]=(bits&one(UInt64))==0 ? a[k] : b[k]; bits>>=1
        end
        i=stop+1
    end
    out
end

# Per-gene Bernoulli(pm) mutation, sampled with geometric skips: O(#mutations) RNG calls
# instead of O(n). Distribution identical to the per-node loop.
function mutate!(genes,f,pm,rng)
    g=f.g; n=g.n
    pm<=0 && return genes
    @inline function flip!(u)
        if g.deg[u]>1
            old=genes[u]; pool=g.adj[u]; j=rand(rng,1:length(pool))
            while pool[j]==old; j=rand(rng,1:length(pool)); end
            genes[u]=pool[j]
        end
    end
    if pm>=1
        for u in 1:n; flip!(u); end
        return genes
    end
    lg=log1p(-pm); u=0
    while true
        s=log(1.0-rand(rng))/lg          # >=0 ; number of non-mutated nodes before the next one
        s>=n && break
        u+=1+floor(Int,s); u>n && break
        flip!(u)
    end
    genes
end

# genes encoding of a labelling: BFS tree inside each community, rooted at its max-degree node.
# O(n+m) (the previous version allocated an n-sized `seen` per community -> O(K*n)).
function reencode(f::LARFactory, labels::Vector{Int})
    g=f.g; n=g.n; K=maximum(labels)
    root=zeros(Int,K); csize=zeros(Int,K)
    @inbounds for u in 1:n
        c=labels[u]; csize[c]+=1
        (root[c]==0 || g.deg[u]>g.deg[root[c]]) && (root[c]=u)
    end
    genes=zeros(Int,n); seen=falses(n); q=Int[]
    for c in 1:K
        csize[c]>=2 || error("Reencode singleton community")
        r=root[c]; seen[r]=true; empty!(q); push!(q,r); firstchild=0; head=1
        while head<=length(q)
            u=q[head]; head+=1
            for v in g.adj[u]
                if labels[v]==c && !seen[v]
                    seen[v]=true; genes[v]=u; firstchild==0 && (firstchild=v); push!(q,v)
                end
            end
        end
        length(q)==csize[c] || error("Reencode disconnected community")
        genes[r]=firstchild
    end
    genes
end
