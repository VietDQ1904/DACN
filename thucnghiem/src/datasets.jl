const DATASETS=("karate","polbooks","football","email","amazon")

function read_text(path)
    endswith(path,".gz") ? read(GzipDecompressorStream(open(path)),String) : read(path,String)
end

function parse_gml(path)
    s=read_text(path); nodes=Int[]; edges=Tuple{Int,Int}[]; labels=Dict{Int,Any}()
    # Tokenise, keeping quoted strings (e.g. team / book titles) as ONE token.
    toks=String[]
    for m in eachmatch(r"\"[^\"]*\"|\[|\]|[^\s\[\]]+",s); push!(toks,m.match); end
    i=1; n=length(toks)
    while i<=n
        t=toks[i]
        if (t=="node" || t=="edge") && i+1<=n && toks[i+1]=="["
            i+=2; kv=Dict{String,String}()
            while i<=n && toks[i]!="]"
                if i+1<=n && toks[i+1]!="[" && toks[i+1]!="]"
                    kv[toks[i]]=toks[i+1]; i+=2
                else; i+=1; end
            end
            if t=="node" && haskey(kv,"id")
                id=parse(Int,kv["id"]); push!(nodes,id)
                for key in ("value","community","group")      # ground truth; never "label" (a name)
                    haskey(kv,key) && (labels[id]=parse_label(kv[key]); break)
                end
            elseif t=="edge" && haskey(kv,"source") && haskey(kv,"target")
                push!(edges,(parse(Int,kv["source"]),parse(Int,kv["target"])))
            end
        end
        i+=1
    end
    ids=sort!(unique(nodes)); mp=Dict(x=>i for (i,x) in enumerate(ids))
    e2=Tuple{Int,Int}[(mp[a],mp[b]) for (a,b) in edges if haskey(mp,a)&&haskey(mp,b)]
    truth=nothing
    if !isempty(labels)
        raw=[get(labels,id,nothing) for id in ids]
        vals=unique(x for x in raw if x!==nothing); cm=Dict(x=>i for (i,x) in enumerate(sort!(collect(vals),by=string)))
        truth=Int[x===nothing ? -1 : cm[x] for x in raw]
    end
    length(ids),e2,truth
end
parse_label(x)=(y=replace(x,"\""=>""); v=tryparse(Int,y); v===nothing ? String(y) : v)

function parse_edges(path)
    lines=split(read_text(path),'\n'); es=Tuple{Int,Int}[]; mx=0
    for line in lines
        st=strip(line); (isempty(st) || startswith(st,"#")) && continue
        p=split(st); length(p)<2 && continue
        a=parse(Int,p[1])+1; b=parse(Int,p[2])+1; push!(es,(a,b)); mx=max(mx,max(a,b))
    end
    mx,es
end

function preprocess_edges(n,edges; truth=nothing,name="",compute_cc=true)
    # isolate removal + largest connected component, matching Python preprocessing.
    _,e2,mp=largest_component(n,edges); t2=nothing
    if truth!==nothing
        old=[0 for _ in 1:length(mp)]
        for (oldid,newid) in mp; old[newid]=truth[oldid]; end
        t2=old
    end
    build_graph(length(mp),e2;truth=t2,name=name,compute_cc=compute_cc)
end

function load_dataset(name::String,data_dir="data")
    if name=="karate"
        # Standard Zachary graph embedded in Graphs.jl.
        # edges from the canonical karate dataset
        pairs=[(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,11),(1,12),(1,13),(1,14),(1,18),(1,20),(1,22),(1,32),(2,3),(2,4),(2,8),(2,14),(2,18),(2,20),(2,22),(2,31),(3,4),(3,8),(3,9),(3,10),(3,14),(3,28),(3,29),(3,33),(4,8),(4,13),(4,14),(5,7),(5,11),(6,7),(6,11),(6,17),(7,17),(9,31),(9,33),(9,34),(10,34),(14,34),(15,33),(15,34),(16,33),(16,34),(19,33),(19,34),(20,34),(21,33),(21,34),(23,33),(23,34),(24,26),(24,28),(24,30),(24,33),(24,34),(25,26),(25,28),(25,32),(26,32),(27,30),(27,34),(28,34),(29,32),(29,34),(30,33),(30,34),(31,33),(31,34),(32,33),(32,34),(33,34)]
        officer=Set([9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34])  # Zachary split
        truth=[u in officer ? 2 : 1 for u in 1:34]
        return build_graph(34,pairs;truth=truth,name="karate")
    elseif name in ("polbooks","football")
        path=joinpath(data_dir,name*".gml"); n,e,t=parse_gml(path); return preprocess_edges(n,e;truth=t,name=name, compute_cc=true)
    elseif name=="polblogs"
        path=joinpath(data_dir,"polblogs.gml"); n,e,t=parse_gml(path); return preprocess_edges(n,e;truth=t,name=name, compute_cc=true)
    elseif name=="email"
        n,e=parse_edges(joinpath(data_dir,"email-Eu-core.txt.gz")); rawlab=Dict{Int,Int}();
        for line in split(read_text(joinpath(data_dir,"email-Eu-core-department-labels.txt.gz")),'\n')
            p=split(strip(line)); length(p)>=2 || continue; rawlab[parse(Int,p[1])+1]=parse(Int,p[2])+1
        end
        truth=[get(rawlab,i,-1) for i in 1:n]; return preprocess_edges(n,e;truth=truth,name="email",compute_cc=true)
    elseif startswith(name,"amazon")
        # Optional syntax amazon:top=K is parsed by the caller; full graph uses all edges.
        n,e=parse_edges(joinpath(data_dir,"com-amazon.ungraph.txt.gz")); return preprocess_edges(n,e;name=name,compute_cc=false)
    else
        error("Unknown dataset: $name")
    end
end
