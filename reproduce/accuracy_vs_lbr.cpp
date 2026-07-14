// accuracy_vs_lbr.cpp -- independent-oracle accuracy of ANNULUS vs Let's Be Rational.
//   reads an oracle file (int64 n; then n x {double h, double c, double v_oracle});
//   for each quote computes annulus sigma and LBR sigma, scores rel error and ULP
//   distance vs the oracle double, broken down by routed chart (WING/LEFT/CENTRAL/RIGHT).
//   Optional: --dump <file> writes "h v_oracle rel_annulus region" rows for the heatmap.
// Build (Linux, with LBR):  see linux_accuracy.sh
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include "lets_be_rational.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
using namespace volfi_annulus;

static uint64_t bitsd(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
// ULP distance between two doubles of the same sign (monotone bit ordering).
static double ulps(double a,double b){
    if(a==b) return 0.0;
    int64_t ia=(int64_t)bitsd(a), ib=(int64_t)bitsd(b);
    if(ia<0) ia=(int64_t)0x8000000000000000ULL - ia;
    if(ib<0) ib=(int64_t)0x8000000000000000ULL - ib;
    int64_t d = ia>ib? ia-ib : ib-ia;
    return (double)d;
}
// LBR normalised implied vol, exactly as benchmark_vec's B path.
static double lbr_sigma(const volfi::otm_context& q,double c){
    double beta=c/q.eh2; return NormalisedImpliedBlackVolatility(beta,-q.h,1.0);
}
// route a quote to its chart tag, mirroring the scalar router (WING has precedence).
static char region_of(double h,double c){
    if(!(c>0.0)||c>=1.0) return 'X';
    if(h==0.0) return 'A';                                   // exact ATM line
    if(c < br::cwing_price(h)) return 'W';                   // WING
    int band; if(detail::grid_central_cell(h,c,detail::bits_of(c),band)>=0) return 'C';
    int rt=detail::grid_endpoint_route(h,c);
    return rt==1?'L': rt==2?'R':'E';                         // LEFT / RIGHT / edge
}
struct Stat{ double maxrel=0, sumsq=0; double maxulp=0; long n=0;
    void add(double rel,double u){ maxrel=std::max(maxrel,rel); sumsq+=rel*rel; maxulp=std::max(maxulp,u); ++n; }
    double rms()const{ return n? std::sqrt(sumsq/n):0.0; } };

int main(int argc,char**argv){
#if defined(VA_SIMD512)
    const char* isa="AVX-512";
#elif defined(VA_SIMD256)
    const char* isa="AVX2+FMA";
#else
    const char* isa="scalar";
#endif
    const char* path = argc>1? argv[1] : "oracle_heat.bin";
    const char* dump = nullptr;
    for(int i=1;i<argc;i++) if(!std::strcmp(argv[i],"--dump")&&i+1<argc) dump=argv[++i];
    FILE* f=fopen(path,"rb"); if(!f){printf("cannot open %s\n",path);return 2;}
    int64_t n; if(fread(&n,8,1,f)!=1)return 2;
    std::vector<double> H(n),C(n),V(n);
    for(int64_t i=0;i<n;++i){double b[3]; if(fread(b,8,3,f)!=3)return 2; H[i]=b[0];C[i]=b[1];V[i]=b[2];}
    fclose(f);

    // per-region stats for ANNULUS and LBR
    Stat aA[128], aL[128];   // indexed by region char
    long over_a=0, over_l=0;
    double worstA=0; int64_t worstAi=-1;
    FILE* df = dump? fopen(dump,"wb") : nullptr;
    for(int64_t i=0;i<n;++i){
        double h=H[i], c=C[i], vo=V[i];
        char rg=region_of(h,c);
        double sa=std::sqrt(implied_variance_otm(h,c));
        volfi::otm_context vq(h);
        double sl=lbr_sigma(vq,c);
        double ra=std::fabs(sa-vo)/std::fabs(vo), ua=ulps(sa,vo);
        double rl=std::fabs(sl-vo)/std::fabs(vo), ul=ulps(sl,vo);
        aA[(int)rg].add(ra,ua); aL[(int)rg].add(rl,ul);
        if(ra>1e-15) ++over_a; if(rl>1e-15) ++over_l;
        if(ra>worstA){worstA=ra;worstAi=i;}
        if(df) std::fprintf(df,"%.17g %.17g %.17e %.17e %c %.17g\n",h,vo,ra,rl,rg,c);
    }
    if(df) fclose(df);

    printf("=== ACCURACY vs independent mpmath oracle  ISA=%s  file=%s  n=%lld ===\n",isa,path,(long long)n);
    printf("  ANNULUS: pts>1e-15 = %ld ;  LBR: pts>1e-15 = %ld\n",over_a,over_l);
    printf("  %-8s | %-30s | %-30s\n","region","ANNULUS  max_rel  rms_rel  max_ulp","LBR      max_rel  rms_rel  max_ulp");
    const char* regs="WLCRAE";
    for(const char* p=regs; *p; ++p){ int r=(int)*p; if(!aA[r].n) continue;
        printf("  [%c] n=%-6ld| %10.3e %10.3e %7.2f      | %10.3e %10.3e %7.2f\n",
               *p,aA[r].n, aA[r].maxrel,aA[r].rms(),aA[r].maxulp,
               aL[r].maxrel,aL[r].rms(),aL[r].maxulp);
    }
    // overall
    Stat gA,gL; for(const char* p=regs;*p;++p){int r=*p;
        gA.maxrel=std::max(gA.maxrel,aA[r].maxrel); gA.maxulp=std::max(gA.maxulp,aA[r].maxulp); gA.sumsq+=aA[r].sumsq; gA.n+=aA[r].n;
        gL.maxrel=std::max(gL.maxrel,aL[r].maxrel); gL.maxulp=std::max(gL.maxulp,aL[r].maxulp); gL.sumsq+=aL[r].sumsq; gL.n+=aL[r].n; }
    printf("  ALL n=%-6ld| %10.3e %10.3e %7.2f      | %10.3e %10.3e %7.2f\n",
           gA.n,gA.maxrel,gA.rms(),gA.maxulp,gL.maxrel,gL.rms(),gL.maxulp);
    if(worstAi>=0) printf("  worst ANNULUS: rel=%.3e at h=%.6g c=%.16g v_or=%.16g region=%c\n",
                          worstA,H[worstAi],C[worstAi],V[worstAi],region_of(H[worstAi],C[worstAi]));
    return 0;
}
