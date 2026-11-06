abstract type Evaluator end

struct CPUEvaluator <: Evaluator
    threaded::Bool
end

CPUEvaluator() = CPUEvaluator(Threads.nthreads() > 1)

function evaluate_batch(
    ev::CPUEvaluator,
    g::GraphData,
    genes::Vector{Vector{Int}}
)
    B = length(genes)

    labs = Vector{Vector{Int}}(undef, B)
    qs = zeros(B)
    ps = zeros(B)

    if ev.threaded
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

    return labs, qs, ps
end