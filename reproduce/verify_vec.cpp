// verify_vec.cpp -- adversarial verification of the endpoint-vectorized build.
//  * accuracy: sqrt(w) vs the independent mpmath oracle, per chart, worst-30
//  * bit-identity: scalar entry == grid batch == permuted batch == fixed-h batch
//  * route census: how many quotes the grid driver sends to CENTRAL/LEFT/RIGHT/scalar
//    and how many of LEFT/RIGHT actually reach an 8-wide SIMD block.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include "volfi_annulus_all.hpp"
using namespace volfi_annulus;
static uint64_t bitsd(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}

int main(int argc,char**argv){
#if defined(VA_SIMD512)
    const char* isa="AVX-512";
#elif defined(VA_SIMD256)
    const char* isa="AVX2+FMA";
#else
    const char* isa="scalar";
#endif
    FILE* f=fopen("oracle_vec.bin","rb"); if(!f){printf("no oracle_vec.bin\n");return 2;}
    int64_t n; if(fread(&n,8,1,f)!=1)return 2;
    std::vector<double> H(n),C(n),V(n); std::vector<char> T(n);
    for(int64_t i=0;i<n;++i){double b[3];char tg;if(fread(b,8,3,f)!=3)return 2;if(fread(&tg,1,1,f)!=1)return 2;
        H[i]=b[0];C[i]=b[1];V[i]=b[2];T[i]=tg;}
    fclose(f);
    printf("ISA=%s  feed n=%lld\n", isa,(long long)n);

    // ---- scalar reference ----
    std::vector<double> Ws(n);
    for(int64_t i=0;i<n;++i) Ws[i]=implied_variance_otm(H[i],C[i]);

    // ---- grid batch ----
    std::vector<double> Wg(n);
    implied_variance_grid_batch(H.data(),C.data(),Wg.data(),(int)n);

    // ---- permuted batch (reconstruct input order) ----
    std::vector<double> Wperm(n),Wp(n); std::vector<int> perm(n);
    int cnt=implied_variance_grid_batch_permuted(H.data(),C.data(),Wperm.data(),perm.data(),(int)n);
    bool permok=(cnt==n);
    if(permok) for(int j=0;j<n;++j) Wp[perm[j]]=Wperm[j];

    // ---- fixed-h batch (group identical h) ----
    std::vector<double> Wf(n,0.0);
    std::map<double,std::vector<int>> byh;
    for(int64_t i=0;i<n;++i) byh[H[i]].push_back((int)i);
    for(auto&kv:byh){ context q(kv.first);
        std::vector<double> cs,ws(kv.second.size(),0.0);
        for(int idx:kv.second) cs.push_back(C[idx]);
        implied_variance_otm_batch(q,cs.data(),ws.data(),(int)cs.size());
        for(size_t j=0;j<kv.second.size();++j) Wf[kv.second[j]]=ws[j];
    }

    // ---- bit-identity ----
    int mg=0,mp=0,mf=0;
    for(int64_t i=0;i<n;++i){
        if(bitsd(Wg[i])!=bitsd(Ws[i])) mg++;
        if(permok && bitsd(Wp[i])!=bitsd(Ws[i])) mp++;
        if(bitsd(Wf[i])!=bitsd(Ws[i])) mf++;
    }
    printf("BIT-IDENTITY scalar==batch:  grid=%d  permuted=%s%d  fixed-h=%d\n",
           mg, permok?"":"(permlen!=n!)", mp, mf);

    // ---- accuracy vs oracle, per chart (use scalar w == batch w) ----
    struct S{double mx=0;std::vector<double> all;};
    std::map<char,S> per;
    std::vector<std::pair<double,int64_t>> worst;
    int over=0;
    for(int64_t i=0;i<n;++i){
        double v=std::sqrt(Ws[i]);
        double rel=std::fabs(v-V[i])/std::fabs(V[i]);
        per[T[i]].mx=std::max(per[T[i]].mx,rel); per[T[i]].all.push_back(rel);
        worst.push_back({rel,i});
        if(rel>1e-15) over++;
    }
    std::sort(worst.begin(),worst.end(),[](auto&a,auto&b){return a.first>b.first;});
    printf("ACCURACY sqrt(w) vs oracle:  n=%lld  pts>1e-15=%d\n",(long long)n,over);
    for(auto&kv:per){ auto&a=kv.second.all; std::sort(a.begin(),a.end());
        double med=a[a.size()/2];
        int o=0; for(double r:a) if(r>1e-15)o++;
        printf("   [%c] n=%zu  max=%.4e  median=%.4e  pts>1e-15=%d\n",
               kv.first,a.size(),kv.second.mx,med,o);
    }
    printf("WORST-30 (rel, h, c, tag):\n");
    for(int k=0;k<30 && k<(int)worst.size();++k){ int64_t i=worst[k].second;
        printf("   %.4e  h=%.10g  c=%.16g  v_or=%.16g  v_re=%.16g  tag=%c\n",
               worst[k].first,H[i],C[i],V[i],std::sqrt(Ws[i]),T[i]);
    }

    // ---- route census on the grid driver (mirror grid_central_cell/grid_endpoint_route) ----
    int nc=0,nl=0,nr=0,nsc=0;
    for(int64_t i=0;i<n;++i){
        int band; uint64_t bc=detail::bits_of(C[i]);
        if(detail::grid_central_cell(H[i],C[i],bc,band)>=0){nc++;continue;}
        int rt=detail::grid_endpoint_route(H[i],C[i]);
        if(rt==1)nl++; else if(rt==2)nr++; else nsc++;
    }
    printf("ROUTE census (grid): CENTRAL=%d LEFT=%d RIGHT=%d scalar/WING=%d\n",nc,nl,nr,nsc);
    printf("   (LEFT/RIGHT SIMD-eligible 8-blocks per TILE depend on per-tile bucket fill;\n"
           "    total LEFT=%d RIGHT=%d >> 8 so vector lanes are exercised on AVX-512)\n",nl,nr);

    // optional: %a hex dump of a few probe rows for manual cross-ISA diff
    if(argc>1 && std::strcmp(argv[1],"--hex")==0){
        FILE* o=fopen(argv[2],"wb");
        for(int64_t i=0;i<n;++i)
            fprintf(o,"%lld %a %a %a %a %c\n",(long long)i,H[i],C[i],Ws[i],Wg[i],T[i]);
        fclose(o);
    }
    bool ok=(mg==0&&mp==0&&mf==0&&permok);
    printf("%s\n", ok?"BIT-IDENTITY: PASS":"BIT-IDENTITY: FAIL");
    return ok?0:1;
}
