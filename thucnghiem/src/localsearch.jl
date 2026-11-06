# ============================================================================
# Local search for MO-HGA-LS  (same algorithm: boundary sweeps, incremental
# Q/phi, per-node Pareto filter, knee move, Pareto archive with crowding).
#
# Engineering changes vs. the reference version:
#   * lock-step multi-source connectivity test (cost ~ smaller side of a cut)
#     + lazy evaluation (only when a move would really act) + per-node cache
#   * boundary maintained incrementally in O(deg(u)) per move through
#     "external-neighbour counters" (no is_boundary rescans, no Set)
#   * archive stores a move JOURNAL, not a full label copy per entry
#     (memory O(#moves) instead of O(n * #insertions))
#   * all scratch buffers live in a reusable LSWorkspace (one per thread),
#     so a call allocates ~nothing besides the returned labels
#   * returned (Q, phi) are recomputed exactly (no incremental drift)
# ============================================================================

mutable struct ConnWS
    seen::Vector{Int}
    own::Vector{Int}
    parent::Vector{Int}
    queues::Vector{Vector{Int}}
    stamp::Int
end
ConnWS(n::Int) = ConnWS(zeros(Int, n), zeros(Int, n), Int[], Vector{Int}[], 0)

@inline function _find!(parent::Vector{Int}, i::Int)
    @inbounds begin
        r = i
        while parent[r] != r; r = parent[r]; end
        while parent[i] != r
            nx = parent[i]; parent[i] = r; i = nx
        end
    end
    return r
end

# Does community `a` stay connected after removing `u`?  (exact)
function connected_without(g::GraphData, labels, a, u, ws::ConnWS)
    adj = g.adj
    ws.stamp += 1
    st = ws.stamp
    seen, own, parent, queues = ws.seen, ws.own, ws.parent, ws.queues

    k = 0
    @inbounds for v in adj[u]
        if v != u && labels[v] == a && seen[v] != st
            k += 1
            seen[v] = st
            own[v] = k
            if k > length(queues)
                push!(queues, Int[])
                push!(parent, k)
            end
            parent[k] = k
            q = queues[k]; empty!(q); push!(q, v)
        end
    end
    k == 0 && return false
    k == 1 && return true
    ncomp = k

    @inbounds while true
        progressed = false
        for i in 1:k
            qi = queues[i]
            isempty(qi) && continue
            progressed = true
            x = pop!(qi)
            for y in adj[x]
                (y == u || labels[y] != a) && continue
                if seen[y] != st
                    seen[y] = st
                    own[y] = i
                    push!(qi, y)
                else
                    ri = _find!(parent, i)
                    rj = _find!(parent, own[y])
                    if ri != rj
                        parent[ri] = rj
                        ncomp -= 1
                        ncomp == 1 && return true
                    end
                end
            end
            if isempty(qi)
                r = _find!(parent, i)
                alive = false
                for t in 1:k
                    if !isempty(queues[t]) && _find!(parent, t) == r
                        alive = true; break
                    end
                end
                alive || return false
            end
        end
        progressed || return false
    end
end

# set LS_CHECK[]=true in tests: verifies archive bookkeeping against exact Q/phi
const LS_CHECK = Ref(false)
# Reference behaviour: if exactly ONE point is non-dominated and it is a move (it dominates the
# current solution) it is archived but NOT applied (`length(nd)==1 && continue`). Set true to apply it.
const APPLY_SINGLE_DOMINATING = Ref(true)

struct ArchEntry
    jlen::Int        # number of journal moves applied before this solution
    u::Int           # extra candidate move (0 = none)
    b::Int
    q::Float64
    phi::Float64
end

# Reusable scratch space. One per thread; never shared between concurrent calls.
mutable struct LSWorkspace
    conn::ConnWS
    ext::Vector{Int}          # #neighbours with a different label
    pos::Vector{Int}          # position in blist (0 = not boundary)
    blist::Vector{Int}
    boundary::Vector{Int}
    c_ver::Vector{Int}        # connectivity cache: version stamp per node
    c_res::Vector{Bool}
    clock::Int                # globally unique community-version generator
    cnt::Vector{Int}
    comm_ver::Vector{Int}
    touched::Vector{Int}
    mb::Vector{Int}; mkub::Vector{Int}
    mq2::Vector{Float64}; mphi2::Vector{Float64}
    mlb2::Vector{Int}; mdb2::Vector{Int}
    pts::Vector{Tuple{Float64,Float64}}
    nd::Vector{Int}; ndkeep::Vector{Bool}
    jr_u::Vector{Int}; jr_b::Vector{Int}
    cd::Vector{Float64}; cord::Vector{Int}      # crowding scratch
end
LSWorkspace(n::Int) = LSWorkspace(ConnWS(n), zeros(Int, n), zeros(Int, n), Int[], Int[],
    zeros(Int, n), fill(false, n), 0, Int[], Int[], Int[],
    Int[], Int[], Float64[], Float64[], Int[], Int[],
    Tuple{Float64,Float64}[], Int[], Bool[], Int[], Int[], Float64[], Int[])

function _prepare!(w::LSWorkspace, n::Int, ncomm::Int)
    if length(w.ext) != n
        w.conn = ConnWS(n); w.ext = zeros(Int, n); w.pos = zeros(Int, n)
        w.c_ver = zeros(Int, n); w.c_res = fill(false, n); empty!(w.blist)
    end
    for x in w.blist; w.pos[x] = 0; end
    empty!(w.blist)
    resize!(w.cnt, ncomm); fill!(w.cnt, 0)
    resize!(w.comm_ver, ncomm)
    @inbounds for c in 1:ncomm; w.comm_ver[c] = (w.clock += 1); end
    empty!(w.jr_u); empty!(w.jr_b)
    return w
end

@inline function _badd!(w::LSWorkspace, x::Int)
    @inbounds if w.pos[x] == 0
        push!(w.blist, x); w.pos[x] = length(w.blist)
    end
end
@inline function _brem!(w::LSWorkspace, x::Int)
    @inbounds begin
        p = w.pos[x]
        if p != 0
            last = w.blist[end]
            w.blist[p] = last; w.pos[last] = p
            pop!(w.blist); w.pos[x] = 0
        end
    end
end

function local_search(g, labels0, i_ls, rng, multi=true, archive_max=20;
                      profiler=nothing, ws::Union{Nothing,LSWorkspace}=nothing)
    n = length(labels0)
    w = ws === nothing ? LSWorkspace(n) : ws
    adj = g.adj; deg = g.deg

    lab = copy(labels0)
    K = maximum(lab)
    l, d = community_stats(g, lab)
    sz = counts(lab)
    q, phi = q_phi_from(g, l, d)

    m = Float64(g.m)
    inv_m = 1.0 / m
    inv_2m = 1.0 / (2.0 * m)
    inv_K = 1.0 / Float64(K)

    _prepare!(w, n, max(length(l), K))
    ext, cnt, comm_ver = w.ext, w.cnt, w.comm_ver
    touched, mb, mkub, mq2, mphi2, mlb2, mdb2 =
        w.touched, w.mb, w.mkub, w.mq2, w.mphi2, w.mlb2, w.mdb2
    pts, ndl, ndkeep, boundary = w.pts, w.nd, w.ndkeep, w.boundary
    jr_u, jr_b = w.jr_u, w.jr_b

    # external-neighbour counters + boundary list: O(m) once
    @inbounds for u in 1:n
        cu = lab[u]; e = 0
        for v in adj[u]
            lab[v] != cu && (e += 1)
        end
        ext[u] = e
        e > 0 && _badd!(w, u)
    end
    isempty(w.blist) && return (labels0, q, phi, false)

    # archive as journal entries
    al = [ArchEntry(0, 0, 0, q, phi)]
    af = [F(q, phi, multi)]
    changed = false

    for _ in 1:i_ls
        isempty(w.blist) && break
        resize!(boundary, length(w.blist)); copyto!(boundary, w.blist)
        shuffle!(rng, boundary)
        al_changed = false
        pass_moved = false

        for u in boundary
            a = lab[u]
            sz[a] >= 3 || continue

            empty!(touched)
            @inbounds for v in adj[u]
                c = lab[v]
                cnt[c] == 0 && push!(touched, c)
                cnt[c] += 1
            end
            kua = cnt[a]
            length(touched) > 1 && sort!(touched)   # candidate order independent of adjacency layout
            if length(touched) <= 1
                @inbounds for c in touched; cnt[c] = 0; end
                continue
            end

            ku = deg[u]
            la = l[a]; da = d[a]
            la2 = la - kua
            da2 = da - ku
            phia  = conductance(g, la,  da)
            phia2 = conductance(g, la2, da2)
            qa  = la  * inv_m - (da  * inv_2m)^2
            qa2 = la2 * inv_m - (da2 * inv_2m)^2

            empty!(mb); empty!(mkub); empty!(mq2); empty!(mphi2); empty!(mlb2); empty!(mdb2)
            @inbounds for b in touched
                b == a && continue
                kub = cnt[b]
                lb = l[b]; db = d[b]
                lb2 = lb + kub
                db2 = db + ku
                qb  = lb  * inv_m - (db  * inv_2m)^2
                qb2 = lb2 * inv_m - (db2 * inv_2m)^2
                push!(mb, b); push!(mkub, kub)
                push!(mq2, q + (qa2 - qa) + (qb2 - qb))
                push!(mphi2, phi + (phia2 + conductance(g, lb2, db2) -
                                    phia - conductance(g, lb, db)) * inv_K)
                push!(mlb2, lb2); push!(mdb2, db2)
            end
            @inbounds for c in touched; cnt[c] = 0; end

            nm = length(mb)
            nm == 0 && continue
            resize!(pts, nm + 1)
            pts[1] = F(q, phi, multi)
            @inbounds for i in 1:nm
                pts[i + 1] = F(mq2[i], mphi2[i], multi)
            end

            nd_indices!(ndl, ndkeep, pts)
            single = length(ndl) == 1
            single && ndl[1] == 1 && continue          # only "stay" is non-dominated

            # lazy validity: connectivity is only tested if something would act
            jk = (single && !APPLY_SINGLE_DOMINATING[]) ? 1 : knee_index_idx(pts, ndl)
            needs = jk != 1
            if !needs
                for j in ndl
                    j == 1 && continue
                    f = pts[j]
                    dom = false
                    @inbounds for x in af
                        if dominates(x, f) || x == f
                            dom = true; break
                        end
                    end
                    if !dom; needs = true; break; end
                end
            end
            needs || continue

            if kua >= 2
                if w.c_ver[u] == comm_ver[a]
                    ok = w.c_res[u]
                else
                    ok = connected_without(g, lab, a, u, w.conn)
                    w.c_ver[u] = comm_ver[a]; w.c_res[u] = ok
                end
                ok || continue
            end

            jl = length(jr_u)
            for j in ndl
                j == 1 && continue
                f = pts[j]
                dominated = false
                @inbounds for x in af
                    if dominates(x, f) || x == f
                        dominated = true; break
                    end
                end
                dominated && continue

                wn = 0
                @inbounds for t in eachindex(af)
                    if !dominates(f, af[t])
                        wn += 1
                        if wn != t
                            af[wn] = af[t]; al[wn] = al[t]
                        end
                    end
                end
                resize!(af, wn); resize!(al, wn)
                push!(al, ArchEntry(jl, u, mb[j - 1], mq2[j - 1], mphi2[j - 1]))
                push!(af, f)
                al_changed = true

                length(al) > archive_max && truncate_crowding!(af, al, archive_max, w.cd, w.cord)
            end

            jk == 1 && continue
            b = mb[jk - 1]; kub = mkub[jk - 1]
            l[a] = la2;  l[b] = mlb2[jk - 1]
            d[a] = da2;  d[b] = mdb2[jk - 1]
            sz[a] -= 1;  sz[b] += 1
            comm_ver[a] = (w.clock += 1); comm_ver[b] = (w.clock += 1)
            q = mq2[jk - 1]; phi = mphi2[jk - 1]
            changed = true
            pass_moved = true

            # O(deg u) boundary maintenance
            @inbounds for v in adj[u]
                lv = lab[v]
                if lv == a
                    ext[v] += 1; _badd!(w, v)
                elseif lv == b
                    ext[v] -= 1; ext[v] == 0 && _brem!(w, v)
                end
            end
            lab[u] = b
            ext[u] = ku - kub
            ext[u] > 0 ? _badd!(w, u) : _brem!(w, u)
            push!(jr_u, u); push!(jr_b, b)
        end

        # A move can be applied even when its point is already represented by,
        # or dominated by, the local archive.  Do not terminate a sweep merely
        # because the archive did not grow: terminate only when neither the
        # archive nor the working partition changed.
        !al_changed && !pass_moved && break
    end

    if LS_CHECK[]
        for u in 1:n
            e = 0
            for v in adj[u]; lab[v] != lab[u] && (e += 1); end
            (ext[u] == e && ((e > 0) == (w.pos[u] != 0))) ||
                error("boundary/ext bookkeeping wrong at node $u: ext=$(ext[u]) true=$e pos=$(w.pos[u])")
        end
        qq, pp = q_phi(g, lab)
        (abs(qq - q) < 1e-9 && abs(pp - phi) < 1e-9) || error("running q/phi drifted: $q,$phi vs $qq,$pp")
    end
    changed || return (labels0, q, phi, false)

    # rebuild the chosen archive solution from the journal
    j = knee_index(af)
    e = al[j]
    out = copy(labels0)
    @inbounds for t in 1:e.jlen; out[jr_u[t]] = jr_b[t]; end
    e.u != 0 && (out[e.u] = e.b)
    out = canonical(out)
    qx, phix = q_phi(g, out)          # exact values, no incremental drift
    if LS_CHECK[] && (abs(qx - e.q) > 1e-9 || abs(phix - e.phi) > 1e-9)
        error("journal/archive mismatch: exact=($qx,$phix) archive=($(e.q),$(e.phi)) jlen=$(e.jlen)")
    end
    return (out, qx, phix, true)
end

function counts(labels)
    c = zeros(Int, maximum(labels))
    @inbounds for x in labels; c[x] += 1; end
    c
end
