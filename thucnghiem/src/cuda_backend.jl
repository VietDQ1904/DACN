# Optional CUDA backend for MO-HGA-LS (batch fitness evaluation on the GPU).
#
# Usage (needs `] add CUDA` in the project; NOT loaded by MOHGALS itself, so the CPU build never needs CUDA):
#     using MOHGALS, CUDA
#     include("src/cuda_backend.jl"); using .MOHGALSCuda
#     ev = GPUEvaluator(g)                      # g::GraphData
#     r  = run_algorithm(g, "MO-HGA-LS", cfg, seed; evaluator=ev)
#
# What runs on the GPU, for a whole offspring batch (n nodes x B individuals) at once:
#   1. decode  : weakly-connected components of u -> genes[u]   (parallel hooking + pointer jumping)
#   2. Q, phi  : community volumes / internal edges via atomics, then per-community reduction
# Variation, selection and local search stay on the CPU (local search is sequential and data-dependent).
module MOHGALSCuda

using CUDA
using ..MOHGALS: GraphData
import ..MOHGALS: Evaluator, evaluate_batch

export GPUEvaluator

struct GPUEvaluator <: Evaluator
    n::Int; m::Int; maxcells::Int
    deg::CuVector{Int32}; eu::CuVector{Int32}; ev::CuVector{Int32}
end
function GPUEvaluator(g::GraphData; maxcells::Int=20_000_000)
    GPUEvaluator(g.n,g.m,maxcells,CuVector{Int32}(g.deg),CuVector{Int32}(g.edges_u),CuVector{Int32}(g.edges_v))
end

const TPB=256

# ---------------- kernels (1-based, column b of every n x B matrix is one individual) ----------------
function k_hook!(parent,genes,flag,n,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N
        b=(i-1)÷n+1; u=(i-1)%n+1
        v=Int(genes[u,b]); pu=parent[u,b]; pv=parent[v,b]
        if pu!=pv
            hi=max(pu,pv); lo=min(pu,pv)
            CUDA.@atomic parent[hi,b]=min(parent[hi,b],lo)   # values only decrease -> terminates
            flag[1]=Int32(1)
        end
    end
    return
end

function k_compress!(parent,flag,n,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N
        b=(i-1)÷n+1; u=(i-1)%n+1
        p=parent[u,b]; pp=parent[p,b]
        if pp!=p; parent[u,b]=pp; flag[1]=Int32(1); end
    end
    return
end

function k_volume!(vol,parent,deg,n,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N
        b=(i-1)÷n+1; u=(i-1)%n+1
        CUDA.@atomic vol[parent[u,b],b]+=deg[u]
    end
    return
end

function k_internal!(intl,parent,eu,ev,m,M)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=M
        b=(i-1)÷m+1; e=(i-1)%m+1
        a=parent[eu[e],b]
        if a==parent[ev[e],b]; CUDA.@atomic intl[a,b]+=Int32(1); end
    end
    return
end

function k_reduce!(acc,vol,intl,n,N,mf)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N
        b=(i-1)÷n+1; r=(i-1)%n+1
        d=Float64(vol[r,b])
        if d>0
            l=Float64(intl[r,b]); m2=2*mf
            phi=1.0
            if l>0
                den=min(d,m2-d)
                phi=den>0 ? (d-2l)/den : 1.0
            end
            CUDA.@atomic acc[b,1]+=l/mf-(d/m2)^2
            CUDA.@atomic acc[b,2]+=phi
            CUDA.@atomic acc[b,3]+=1.0
        end
    end
    return
end

# ---------------- host side ----------------
function fast_canonical(col::AbstractVector{Int32},n)
    mp=zeros(Int32,n); out=Vector{Int}(undef,n); k=Int32(0)
    @inbounds for u in 1:n
        r=col[u]
        if mp[r]==0; k+=Int32(1); mp[r]=k; end
        out[u]=mp[r]
    end
    out
end

function eval_chunk(ev::GPUEvaluator,G::Matrix{Int32})
    n=ev.n; B=size(G,2); N=n*B
    genes=CuArray(G); parent=CuArray(repeat(Int32.(1:n),1,B)); flag=CUDA.zeros(Int32,1)
    blocks=cld(N,TPB)
    while true
        fill!(flag,0); @cuda threads=TPB blocks=blocks k_hook!(parent,genes,flag,n,N)
        Array(flag)[1]==0 && break
        while true                                   # pointer jumping until fully compressed
            fill!(flag,0); @cuda threads=TPB blocks=blocks k_compress!(parent,flag,n,N)
            Array(flag)[1]==0 && break
        end
    end
    vol=CUDA.zeros(Int32,n,B); intl=CUDA.zeros(Int32,n,B); acc=CUDA.zeros(Float64,B,3)
    @cuda threads=TPB blocks=blocks k_volume!(vol,parent,ev.deg,n,N)
    M=ev.m*B
    @cuda threads=TPB blocks=cld(M,TPB) k_internal!(intl,parent,ev.eu,ev.ev,ev.m,M)
    @cuda threads=TPB blocks=blocks k_reduce!(acc,vol,intl,n,N,Float64(ev.m))
    A=Array(acc); P=Array(parent)
    CUDA.unsafe_free!(genes); CUDA.unsafe_free!(parent); CUDA.unsafe_free!(vol); CUDA.unsafe_free!(intl); CUDA.unsafe_free!(acc)
    labs=[fast_canonical(view(P,:,b),n) for b in 1:B]
    qs=Vector{Float64}(undef,B); ps=Vector{Float64}(undef,B)
    for b in 1:B
        K=A[b,3]
        if K<=1; qs[b]=-0.5; ps[b]=1.0 else qs[b]=A[b,1]; ps[b]=A[b,2]/K end
    end
    labs,qs,ps
end

function evaluate_batch(ev::GPUEvaluator,g::GraphData,genes::Vector{Vector{Int}})
    B=length(genes); chunk=max(1,min(B,ev.maxcells÷ev.n))
    labs=Vector{Vector{Int}}(undef,B); qs=zeros(B); ps=zeros(B)
    for s in 1:chunk:B
        e=min(B,s+chunk-1); G=Matrix{Int32}(undef,ev.n,e-s+1)
        for (j,i) in enumerate(s:e); @inbounds G[:,j]=genes[i]; end
        l,q,p=eval_chunk(ev,G)
        labs[s:e]=l; qs[s:e]=q; ps[s:e]=p
    end
    labs,qs,ps
end

end # module
