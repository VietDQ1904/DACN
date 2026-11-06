const ALGOS=("GA","HGA","NSGA2","MO-HGA-LS")

# ---- population evaluators: decode genes -> labels and compute (Q, phi) for a whole batch ----
abstract type Evaluator end
struct CPUEvaluator <: Evaluator; threaded::Bool; end
CPUEvaluator()=CPUEvaluator(Threads.nthreads()>1)
function evaluate_batch(ev::CPUEvaluator,g::GraphData,genes::Vector{Vector{Int}})
    B=length(genes); labs=Vector{Vector{Int}}(undef,B); qs=zeros(B); ps=zeros(B)
    # Thread scheduling is not free. For small graphs/batches it can cost more
    # than decoding and evaluating the individuals; use a workload threshold.
    if ev.threaded && B * max(g.m, 1) >= 100_000
        Threads.@threads for i in 1:B
            labs[i] = decode(g, genes[i])
            qs[i], ps[i] = q_phi(g, labs[i])
        end
    else
        for i in 1:B
            labs[i] = decode(g, genes[i])
            qs[i], ps[i] = q_phi(g, labs[i])
        end
    end
    labs,qs,ps
end

Base.@kwdef mutable struct Config
    mu::Int=100; lam::Int=100; pc::Float64=0.85; pm::Float64=0.0; tournament::Int=2
    i_ls::Int=30; p_ls::Float64=0.2; rho::Float64=0.7; a_max::Int=100
    g_max::Int=200; k_stag::Int=30; eps_stop::Float64=1e-4
end

mutable struct RunResult
    dataset::String; algo::String; seed::Int; q::Float64; phi::Float64; nmi::Float64; ari::Float64
    k::Int; runtime::Float64; generations::Int; n_eval::Int; archive_size::Int; hv_final::Float64
    hv_history::Vector{Float64}; q_history::Vector{Float64}; time_history::Vector{Float64}
    ls_calls::Int; ls_changed::Int; ls_improved::Int; ls_time::Float64; ls_dq::Float64; ls_dphi::Float64
    labels::Vector{Int}
end

# hashes of many label vectors, in parallel (O(N*n))
function thashes(lists)
    h=Vector{UInt64}(undef,length(lists))
    if Threads.nthreads()>1 && length(lists)>1
        Threads.@threads for i in eachindex(lists); h[i]=hash(lists[i]); end
    else
        for i in eachindex(lists); h[i]=hash(lists[i]); end
    end
    h
end

function dedup_partition(labels_list)
    hs=thashes(labels_list); seen=Set{UInt64}(); keep=Int[]
    for i in eachindex(labels_list)
        !(hs[i] in seen) && (push!(seen,hs[i]); push!(keep,i))
    end
    keep
end

function archive_update(archive, labels, qs, phis, multi, amax)
    pool=copy(archive); seen=Set{UInt64}(thashes([x[1] for x in pool])); hs=thashes(labels)
    for i in eachindex(labels)
        h=hs[i]; h in seen && continue
        push!(seen,h); push!(pool,(copy(labels[i]),qs[i],phis[i]))
    end
    P=[F(x[2],x[3],multi) for x in pool]; nd=nd_indices(P); pool=pool[nd]; P=P[nd]
    if length(pool)>amax; kp=truncate_crowding(P,amax); pool=pool[kp]; end
    pool
end

# ---- parallel variation: one independent RNG stream per child (deterministic for any #threads) ----
function make_offspring(f,cfg,genes,p1,p2,pm,rng)
    lam=cfg.lam; seeds=rand(rng,UInt64,lam); out=Vector{Vector{Int}}(undef,lam)
    body=i->begin
        r=Random.Xoshiro(seeds[i])
        child=rand(r)<cfg.pc ? crossover(genes[p1[i]],genes[p2[i]],r) : copy(genes[p1[i]])
        mutate!(child,f,pm,r); out[i]=child
    end
    # Avoid spawning workers for tiny offspring batches.
    if Threads.nthreads() > 1 && lam * max(f.g.n, 1) >= 50_000
        Threads.@threads for i in 1:lam
            body(i)
        end
    else
        for i in 1:lam
            body(i)
        end
    end
    out
end

# ---- parallel local search over the offspring batch -------------------------------------------
# Which children get LS and each child's RNG seed are drawn sequentially from the master RNG, so
# results are reproducible and independent of the number of threads. Workers pull jobs from an
# atomic counter (dynamic load balancing: LS cost varies a lot between children), each with its own
# reusable LSWorkspace. Returns (calls, changed, improved, dq, dphi) and updates the batch in place.
struct LSJob; i::Int; seed::UInt64; end
struct LSOut; nl::Vector{Int}; nq::Float64; np::Float64; chg::Bool; genes::Vector{Int}; end

make_ls_pool(g::GraphData)=[LSWorkspace(g.n) for _ in 1:max(1,Threads.nthreads())]

function ls_batch!(g,f,cfg,rng,multi,use_ls,pool,offgenes,offlabels,oq,op)
    stats=(calls=0,changed=0,improved=0,dq=0.0,dp=0.0)
    use_ls || return stats
    jobs=LSJob[]
    for i in eachindex(offlabels)
        if rand(rng)<cfg.p_ls && maximum(offlabels[i])>=2
            push!(jobs,LSJob(i,rand(rng,UInt64)))
        end
    end
    N=length(jobs); N==0 && return stats
    res=Vector{LSOut}(undef,N)
    run_job=(k,w)->begin
        j=jobs[k]; r=Random.Xoshiro(j.seed)
        nl,nq,np,chg=local_search(g,offlabels[j.i],cfg.i_ls,r,multi,20; ws=w)
        res[k]=LSOut(nl,nq,np,chg, chg ? reencode(f,nl) : Int[])
    end
    nw=min(length(pool),N)
    # Parallel LS pays off only when there is enough independent work.
    # This keeps small benchmarks (Karate/Polbooks/Football) from paying
    # task/atomic scheduling overhead while retaining parallelism on Email.
    parallel_ls = nw > 1 && N >= max(4, 2nw) && g.n * N >= 100_000
    if !parallel_ls
        for k in 1:N
            run_job(k, pool[1])
        end
    else
        ctr=Threads.Atomic{Int}(0); wctr=Threads.Atomic{Int}(0)
        @sync for _ in 1:nw
            Threads.@spawn begin
                w=pool[Threads.atomic_add!(wctr,1)+1]
                while true
                    k=Threads.atomic_add!(ctr,1)+1
                    k>N && break
                    run_job(k,w)
                end
            end
        end
    end
    changed=0; improved=0; dq=0.0; dp=0.0
    for k in 1:N
        o=res[k]; i=jobs[k].i
        if o.chg
            q=oq[i]; p=op[i]; changed+=1; dq+=o.nq-q; dp+=o.np-p
            improved += dominates(F(o.nq,o.np,multi),F(q,p,multi)) ? 1 : 0
            offgenes[i]=o.genes; offlabels[i]=o.nl; oq[i]=o.nq; op[i]=o.np
        end
    end
    (calls=N,changed=changed,improved=improved,dq=dq,dp=dp)
end

function select_representative(archive,multi)
    qs=[x[2] for x in archive]; ps=[x[3] for x in archive]
    iQ=argmax(qs); out=Dict("maxQ"=>archive[iQ])
    if multi
        out["minPhi"]=archive[argmin(ps)]
        P=[F(qs[i],ps[i],true) for i in eachindex(qs)]; out["knee"]=archive[knee_index(P)]
    end
    out
end

function run_algorithm(g::GraphData,algo::String,cfg::Config,seed::Int; evaluator::Evaluator=CPUEvaluator())
    @assert algo in ALGOS
    multi=algo in ("NSGA2","MO-HGA-LS"); use_ls=algo in ("HGA","MO-HGA-LS") && cfg.p_ls>0
    rng=MersenneTwister(seed); f=LARFactory(g,cfg.rho); lspool=use_ls ? make_ls_pool(g) : LSWorkspace[]; pm=cfg.pm>0 ? cfg.pm : 1/g.n
    t0=time(); genes=init_population(f,cfg.mu,rng)
    labels,qs,ps=evaluate_batch(evaluator,g,genes); n_eval=cfg.mu
    P=[F(qs[i],ps[i],multi) for i in 1:cfg.mu]; rank,crowd=rank_crowding(P)
    archive=archive_update(Tuple{Vector{Int},Float64,Float64}[],labels,qs,ps,multi,cfg.a_max)
    hv=hypervolume([F(x[2],x[3],multi) for x in archive]); hvbest=hv; stag=0; gen=0
    hvh=[hv]; qh=[maximum(x[2] for x in archive)]; th=[time()-t0]
    ls_calls=0; ls_changed=0; ls_improved=0; ls_time=0.0; ls_dq=0.0; ls_dp=0.0
    while gen<cfg.g_max && stag<cfg.k_stag
        # tournament selection
        p1=Vector{Int}(undef,cfg.lam); p2=similar(p1)
        for arr in (p1,p2)
            for i in 1:cfg.lam
                best=rand(rng,1:cfg.mu)
                for _ in 2:cfg.tournament
                    c=rand(rng,1:cfg.mu)
                    better=rank[c]<rank[best] || (rank[c]==rank[best] && crowd[c]>crowd[best])
                    better && (best=c)
                end
                arr[i]=best
            end
        end
        # 1) variation (cheap, CPU)  2) batch fitness evaluation (CPU or GPU)  3) local search
        offgenes=make_offspring(f,cfg,genes,p1,p2,pm,rng)
        offlabels,oq,op=evaluate_batch(evaluator,g,offgenes); n_eval+=cfg.lam
        t_ls=time(); st=ls_batch!(g,f,cfg,rng,multi,use_ls,lspool,offgenes,offlabels,oq,op); ls_time+=time()-t_ls
        ls_calls+=st.calls; ls_changed+=st.changed; ls_improved+=st.improved; ls_dq+=st.dq; ls_dp+=st.dp
        Rg=vcat(genes,offgenes); Rl=vcat(labels,offlabels); Rq=vcat(qs,oq); Rp=vcat(ps,op)
        keep=dedup_partition(Rl)
        if length(keep)<cfg.mu
            need=cfg.mu-length(keep); newg=init_population(f,need,rng)
            nl_,nq_,np_=evaluate_batch(evaluator,g,newg); n_eval+=need
            append!(Rg,newg); append!(Rl,nl_); append!(Rq,nq_); append!(Rp,np_)
            keep=collect(eachindex(Rg))
        end
        Rg=Rg[keep]; Rl=Rl[keep]; Rq=Rq[keep]; Rp=Rp[keep]
        RF=[F(Rq[i],Rp[i],multi) for i in eachindex(Rq)]; sel,rank,crowd=nsga_select(RF,cfg.mu)
        genes=Rg[sel]; labels=Rl[sel]; qs=Rq[sel]; ps=Rp[sel]
        archive=archive_update(archive,offlabels,oq,op,multi,cfg.a_max)
        hv=hypervolume([F(x[2],x[3],multi) for x in archive]);
        if hv>hvbest+cfg.eps_stop; hvbest=hv; stag=0; else; stag+=1; end
        push!(hvh,hv); push!(qh,maximum(x[2] for x in archive)); push!(th,time()-t0); gen+=1
    end
    reps=select_representative(archive,multi); chosen=reps["maxQ"]; q,phi=chosen[2],chosen[3]; lab=chosen[1]
    nmi=truth_nmi(g.truth,lab); ari=truth_ari(g.truth,lab)
    RunResult(g.name,algo,seed,q,phi,nmi,ari,maximum(lab),time()-t0,gen,n_eval,length(archive),hv,hvh,qh,th,ls_calls,ls_changed,ls_improved,ls_time,ls_dq,ls_dp,lab)
end
