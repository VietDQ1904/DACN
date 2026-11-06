const HV_REF=(1.1,1.1)
@inline F(q,phi,multi)=multi ? (1-q,phi) : (1-q,0.0)
@inline function dominates(a,b); (a[1]<=b[1] && a[2]<=b[2]) && (a[1]<b[1] || a[2]<b[2]); end

function nd_indices(P)
    n=length(P); keep=trues(n)
    for i in 1:n
        keep[i] || continue
        for j in 1:n
            i==j && continue
            if dominates(P[j],P[i]); keep[i]=false; break; end
        end
    end
    findall(keep)
end

function nd_sort(P)
    n=length(P); rank=fill(-1,n); domcnt=zeros(Int,n); S=[Int[] for _ in 1:n]
    for i in 1:n, j in 1:n
        i==j && continue
        if dominates(P[i],P[j]); push!(S[i],j)
        elseif dominates(P[j],P[i]); domcnt[i]+=1 end
    end
    front=findall(==(0),domcnt); r=0
    while !isempty(front)
        for i in front; rank[i]=r; end
        nxt=Int[]
        for i in front
            for j in S[i]
                domcnt[j]-=1; domcnt[j]==0 && push!(nxt,j)
            end
        end
        front=nxt; r+=1
    end
    rank
end

function crowding(P)
    n=length(P); d=zeros(Float64,n); n<=2 && (d .= Inf; return d)
    for k in 1:2
        ord=sortperm(1:n,by=i->P[i][k]); d[ord[1]]=Inf; d[ord[end]]=Inf
        lo=P[ord[1]][k]; hi=P[ord[end]][k]; span=hi-lo
        span<=0 && continue
        for t in 2:n-1; d[ord[t]] += (P[ord[t+1]][k]-P[ord[t-1]][k])/span; end
    end
    d
end

function rank_crowding(P)
    r=nd_sort(P); c=zeros(Float64,length(P))
    for rr in 0:maximum(r)
        idx=findall(==(rr),r); c[idx].=crowding(P[idx])
    end
    r,c
end

function nsga_select(P,μ)
    r,c=rank_crowding(P); ord=sort(collect(eachindex(P)),by=i->(r[i],-c[i])); sel=ord[1:μ]
    return sel,r[sel],c[sel]
end

function hypervolume(P)
    isempty(P) && return 0.0
    idx=[i for i in eachindex(P) if P[i][1]<HV_REF[1] && P[i][2]<HV_REF[2]]
    sort!(idx,by=i->P[i][1]); hv=0.0; best=HV_REF[2]
    for i in idx
        f1,f2=P[i]
        if f2<best; hv+=(HV_REF[1]-f1)*(best-f2); best=f2; end
    end
    hv
end

function truncate_crowding(P,size)
    keep=collect(eachindex(P))
    while length(keep)>size
        c=crowding(P[keep]); j=argmin([isinf(x) ? Inf : x for x in c]); deleteat!(keep,j)
    end
    keep
end

# Tie-break for equally-distant knee candidates (always happens with exactly 2 non-dominated
# points). The original code resolved it by Dict/hash order, i.e. arbitrarily. Now explicit:
#   :q   -> prefer higher modularity (default; matches the maxQ representative reported by the engine)
#   :phi -> prefer lower conductance
const KNEE_TIE = Ref(:q)
@inline _tie_better(x, y) = KNEE_TIE[] === :phi ? x[2] < y[2] : x[1] < y[1]

function knee_index(P)
    n=length(P); n==1 && return 1
    lo=[minimum(x[1] for x in P),minimum(x[2] for x in P)]; hi=[maximum(x[1] for x in P),maximum(x[2] for x in P)]
    best=1; bd=Inf
    for i in 1:n
        s1=hi[1]-lo[1]; s2=hi[2]-lo[2]
        z1=s1>1e-12 ? (P[i][1]-lo[1])/s1 : 0.0; z2=s2>1e-12 ? (P[i][2]-lo[2])/s2 : 0.0; d=z1*z1+z2*z2
        # exact ties are common (e.g. 2 non-dominated points are always equidistant after
        # normalisation) -> break them deterministically by higher modularity, never by storage order
        (d<bd-1e-12 || (d<=bd+1e-12 && _tie_better(P[i],P[best]))) && (bd=min(d,bd); best=i)
    end
    best
end

# ---- allocation-free helpers used by the local search (same results as the originals) ----
function nd_indices!(out::Vector{Int}, keep::Vector{Bool}, P)
    n=length(P); resize!(keep,n); fill!(keep,true); empty!(out)
    @inbounds for i in 1:n
        for j in 1:n
            i==j && continue
            if dominates(P[j],P[i]); keep[i]=false; break; end
        end
        keep[i] && push!(out,i)
    end
    out
end

# == knee_index(P[idx]) but returns the index INTO P and allocates nothing
function knee_index_idx(P, idx)
    length(idx)==1 && return idx[1]
    l1=Inf; l2=Inf; h1=-Inf; h2=-Inf
    @inbounds for i in idx
        x=P[i]; l1=min(l1,x[1]); h1=max(h1,x[1]); l2=min(l2,x[2]); h2=max(h2,x[2])
    end
    s1=h1-l1; s2=h2-l2; best=idx[1]; bd=Inf
    @inbounds for i in idx
        z1=s1>1e-12 ? (P[i][1]-l1)/s1 : 0.0; z2=s2>1e-12 ? (P[i][2]-l2)/s2 : 0.0
        dd=z1*z1+z2*z2
        (dd<bd-1e-12 || (dd<=bd+1e-12 && _tie_better(P[i],P[best]))) && (bd=min(dd,bd); best=i)
    end
    best
end

# Index (into P) of the point truncate_crowding would delete next: smallest crowding distance,
# boundary points = Inf, first index wins ties. Allocation-free (buffers d, ord are reused);
# stable insertion sort is fine because archives are tiny (<= archive_max+1 points).
function crowd_argmin!(P, d::Vector{Float64}, ord::Vector{Int})
    n=length(P); n<=2 && return 1
    resize!(d,n); resize!(ord,n); fill!(d,0.0)
    @inbounds for k in 1:2
        for i in 1:n; ord[i]=i; end
        for i in 2:n
            x=ord[i]; key=P[x][k]; j=i-1
            while j>=1 && P[ord[j]][k]>key; ord[j+1]=ord[j]; j-=1; end
            ord[j+1]=x
        end
        d[ord[1]]=Inf; d[ord[n]]=Inf
        span=P[ord[n]][k]-P[ord[1]][k]
        span<=0 && continue
        for t in 2:n-1; d[ord[t]]+=(P[ord[t+1]][k]-P[ord[t-1]][k])/span; end
    end
    best=1; bv=d[1]
    @inbounds for i in 2:n; d[i]<bv && (bv=d[i]; best=i); end
    best
end

# in-place truncation of two parallel archives (fitness tuples + payload), same rule as truncate_crowding
function truncate_crowding!(af, al, size::Int, d::Vector{Float64}, ord::Vector{Int})
    while length(af)>size
        j=crowd_argmin!(af,d,ord); deleteat!(af,j); deleteat!(al,j)
    end
    nothing
end
