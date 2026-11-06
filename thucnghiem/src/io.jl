function clean_nan(x)
    if x isa AbstractFloat && isnan(x)
        return nothing
    elseif x isa AbstractDict
        return Dict(k => clean_nan(v) for (k, v) in x)
    elseif x isa AbstractArray
        return [clean_nan(v) for v in x]
    else
        return x
    end
end

function run_to_dict(r::RunResult)
    Dict("dataset"=>r.dataset,"algo"=>r.algo,"seed"=>r.seed,"q"=>r.q,"phi"=>r.phi,"nmi"=>r.nmi,"ari"=>r.ari,"k"=>r.k,"runtime"=>r.runtime,"generations"=>r.generations,"n_eval"=>r.n_eval,"archive_size"=>r.archive_size,"hv_final"=>r.hv_final,"ls_calls"=>r.ls_calls,"ls_changed"=>r.ls_changed,"ls_improved"=>r.ls_improved,"ls_time"=>r.ls_time,"ls_dq"=>r.ls_dq,"ls_dphi"=>r.ls_dphi)
end

function save_runs(path,results)
    mkpath(dirname(path)); open(path,"w") do io
        write(io, "[\n")
        for r in results
            d = run_to_dict(r)
            cleaned_d = clean_nan(d)
            JSON3.write(io, cleaned_d)
            write(io, ",\n")
        end
        write(io, "]")
    end
end

function save_table(path,df)
    mkpath(dirname(path)); CSV.write(path,df)
end
