function contingency(a,b)
    valid=[i for i in eachindex(a) if a[i]>=0 && b[i]>=0];
    va=unique(a[valid]); vb=unique(b[valid]); ia=Dict(x=>i for (i,x) in enumerate(va)); ib=Dict(x=>i for (i,x) in enumerate(vb)); C=zeros(Int,length(va),length(vb))
    for i in valid; C[ia[a[i]],ib[b[i]]]+=1; end
    C
end
function truth_nmi(truth,pred)
    truth===nothing && return NaN; C=contingency(truth,pred); n=sum(C); n==0 && return NaN
    ai=sum(C,dims=2); bj=sum(C,dims=1); mi=0.0
    for i in axes(C,1), j in axes(C,2)
        C[i,j]>0 && (mi += (C[i,j]/n)*log((C[i,j]*n)/(ai[i]*bj[j])))
    end
    # sklearn-style arithmetic mean of entropies
    Ha=-sum((x/n)*log(x/n) for x in ai if x>0); Hb=-sum((x/n)*log(x/n) for x in bj if x>0)
    (Ha+Hb)==0 ? 1.0 : 2mi/(Ha+Hb)
end
function comb2(x); x<2 ? 0.0 : x*(x-1)/2; end
function truth_ari(truth,pred)
    truth===nothing && return NaN; C=contingency(truth,pred); n=sum(C); n<2 && return NaN
    s=sum(comb2.(C)); a=sum(comb2.(vec(sum(C,dims=2)))); b=sum(comb2.(vec(sum(C,dims=1)))); total=comb2(n)
    expected=a*b/total; denom=0.5*(a+b)-expected; denom==0 ? 1.0 : (s-expected)/denom
end
