using Pkg; Pkg.activate(joinpath(@__DIR__, "..")); using MOHGALS, Random, Test

@testset "Dataset" begin
    g=load_dataset("karate","data"); @test (g.n,g.m)==(34,78); @test all(length(g.adj[i])==g.deg[i] for i in 1:g.n)
end
@testset "LAR" begin
    g=load_dataset("karate","data"); f=LARFactory(g,0.7); rng=MersenneTwister(1); x=random_individual(f,rng)
    @test all(any(==(x[i]),g.adj[i]) for i in 1:g.n)
    lab=decode(g,x); @test maximum(lab)>=1; q,p=q_phi(g,lab); @test -1<=q<=1 && -1<=p<=1
end

@testset "Evolution" begin
    g=load_dataset("karate","data"); cfg=Config(mu=10,lam=10,g_max=3,k_stag=3,i_ls=2,p_ls=0.2); r=run_algorithm(g,"MO-HGA-LS",cfg,0)
    @test r.generations<=3; @test r.k>=2; @test isfinite(r.q); @test isfinite(r.phi)
end
println("All core checks passed")
