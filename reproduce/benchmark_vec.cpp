// benchmark_vec.cpp -- Phase-8 BROAD-DOMAIN benchmark for the volfi-annulus
// implied-variance inverter (v up to 8, |x|=h up to 16).
//
// Phase-8 vectorizes the LEFT (matched small-h) and RIGHT (v>2 endpoint) charts
// in the batch drivers.  In Phase-7 those two charts dropped to the scalar
// fallback inside batch (so batch D ~= scalar C for LEFT/RIGHT); here they run
// 8-wide, so this benchmark exposes the per-chart scalar->batch speedup on the
// covered range of EACH chart (no blending across charts).
//
// Methods:
//   A volfi::implied_variance_otm                 (baseline, Halley)
//   B LBR NormalisedImpliedBlackVolatility        (Jaeckel Let's Be Rational)
//   C volfi_annulus::implied_variance_otm         (Phase-8 scalar entry)
//   D volfi_annulus::implied_variance_grid_batch  (mixed-h sort-then-batch)
//   D volfi_annulus::implied_variance_otm_batch   (fixed-h surface; SIMD kernels)
//
// Reported RESTRICTED TO THE COVERED RANGE per chart (CENTRAL / LEFT / RIGHT),
// scalar and batch, on fixed-h surfaces (incl. LEFT-heavy and RIGHT-heavy) and a
// mixed broad grid, plus the broad-domain CHART-COVERAGE fraction.
//
// Build (from this dir, LBR sources on the include/link line):
//   LBR=".../LetsBeRational"
//   g++ -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math -I"$LBR" \
//     benchmark_vec.cpp "$LBR/lets_be_rational.cpp" "$LBR/erf_cody.cpp" \
//     "$LBR/normaldistribution.cpp" "$LBR/rationalcubic.cpp" -o benchmark_vec
//   taskset -c 2 nice -n -5 ./benchmark_vec
// AVX2 build: add -mno-avx512f.  no-FMA: -mavx2 -mno-avx512f.
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include "lets_be_rational.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
namespace {
constexpr std::uint64_t kSeed = 20260710ULL;
constexpr double kSqrtTwoPi = 2.50662827463100050241576528481104525;
volatile double g_sink = 0.0;
int env_int(const char* n,int f){ if(const char*r=std::getenv(n)){try{int v=std::stoi(r); if(v>0)return v;}catch(...){}} return f; }

double qnorm0(double p){
  const double a1=-3.969683028665376e+01,a2=2.209460984245205e+02,a3=-2.759285104469687e+02,
    a4=1.383577518672690e+02,a5=-3.066479806614716e+01,a6=2.506628277459239e+00;
  const double b1=-5.447609879822406e+01,b2=1.615858368580409e+02,b3=-1.556989798598866e+02,
    b4=6.680131188771972e+01,b5=-1.328068155288572e+01;
  const double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,c3=-2.400758277161838e+00,
    c4=-2.549732539343734e+00,c5=4.374664141464968e+00,c6=2.938163982698783e+00;
  const double d1=7.784695709041462e-03,d2=3.224671290700398e-01,d3=2.445134137142996e+00,d4=3.754408661907416e+00;
  double q,r;
  if(p<0.02425){ q=std::sqrt(-2.0*std::log(p));
    return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1.0); }
  if(p<=0.97575){ q=p-0.5; r=q*q;
    return ((((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q)/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0); }
  q=std::sqrt(-2.0*std::log1p(-p));
  return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
}
double qnorm2(double p){ double x=qnorm0(p);
  for(int i=0;i<2;i++){ double e=volfi::phi_cdf(x)-p; double u=e*kSqrtTwoPi*std::exp(0.5*x*x); x-=u/(1.0+0.5*x*u);} return x; }

// ---- Phase-7 broad-domain route label (mirrors implied_variance_otm exactly) --
enum { R_WING=0, R_LEFT=1, R_CENTRAL=2, R_RIGHT=3, R_EDGE=4 };
int route(double h,double c){
  using namespace volfi_annulus;
  if(!(c>0.0)||c>=1.0) return R_EDGE;
  if(h==0.0) return R_EDGE;
  double cw=br::cwing_price(h);
  if(c<cw) return R_WING;
  if(h<H_ATM_HI){
    double ct_left=1.0-br::onem_otm(h,volfi_annulus_broadrange::LEFT_RIGHT_VSEAM,std::exp(h));
    return (c<=ct_left)?R_LEFT:R_RIGHT;
  }
  double ct2=br::c2_price(h);
  if(h<=H_BOX && c<=ct2) return R_CENTRAL;
  return R_RIGHT;
}
// OLD phase-6 in-box fast-path predicate: box table (h in [H_ATM_HI,H_BOX], v<=2,
// c>=cw) OR wing.  Everything else (v>2 in/out of box, small-h deep corner,
// h>H_BOX) fell to a crude analytic clamp/fallback in the OLD architecture.
bool old_covered(double h,double c){
  using namespace volfi_annulus;
  int r=route(h,c);
  if(r==R_WING) return true;                       // wing existed in phase-6
  if(r==R_CENTRAL) return true;                    // in-box main table
  return false;                                    // LEFT/RIGHT/edge were crude
}

static inline uint64_t bd(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }

struct Grid { std::vector<double> h,c,vt; std::vector<volfi::otm_context> q; std::vector<volfi_annulus::context> a; };

// feasibility filter shared with the oracle: 0<c<1 and c not rounding to 0/1.
bool feasible(double c){ return (c>0.0)&&(c<1.0)&&(1.0-c>1e-16); }

double lbr_var(const volfi::otm_context& q,double c){ double beta=c/q.eh2;
  double s=NormalisedImpliedBlackVolatility(beta,-q.h,1.0); return s*s; }

template<class F> double bench(int n,int rep,F&& fn){
  double s0=0,s1=0,s2=0,s3=0; auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<rep;r++){ int i=0; for(;i+3<n;i+=4){ s0+=fn(i); s1+=fn(i+1); s2+=fn(i+2); s3+=fn(i+3);} for(;i<n;i++) s0+=fn(i); }
  auto t1=std::chrono::steady_clock::now(); g_sink+=s0+s1+s2+s3;
  return 1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*n);
}
double median(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
  return n?(n&1?v[n/2]:0.5*(v[n/2-1]+v[n/2])):0.0; }
double quant(std::vector<double> v,double p){ std::sort(v.begin(),v.end()); if(v.empty())return 0;
  double pos=p*(v.size()-1); size_t lo=(size_t)std::floor(pos),hi=(size_t)std::ceil(pos); double w=pos-lo; return v[lo]*(1-w)+v[hi]*w; }
void rpt(const char* g,const char* nm,std::vector<double>& s){
  std::printf("TIMING %-8s %-22s median=%8.2f  IQR=[%8.2f,%8.2f]  ns/eval\n",g,nm,median(s),quant(s,0.25),quant(s,0.75)); }

// build contexts for a Grid
void build(Grid& g){ g.q.clear(); g.a.clear(); g.q.reserve(g.h.size()); g.a.reserve(g.h.size());
  for(double hh:g.h){ g.q.emplace_back(hh); g.a.emplace_back(hh);} }

// subset of grid for one route label
Grid subset(const Grid& G,int rlabel){ Grid s; for(size_t i=0;i<G.h.size();++i){ if(route(G.h[i],G.c[i])==rlabel){ s.h.push_back(G.h[i]); s.c.push_back(G.c[i]); s.vt.push_back(G.vt[i]); } } build(s); return s; }

void time_chart(const char* nm,const Grid& s,int runs,int rep){
  if(s.h.empty()){ std::printf("  %-8s : (no points)\n",nm); return; }
  int n=(int)s.h.size(); std::vector<double> tA,tB,tC,tD; std::vector<double> wbuf(n);
  for(int r=0;r<runs;r++){
    tA.push_back(bench(n,rep,[&](int i){return volfi::implied_variance_otm(s.q[i],s.c[i]);}));
    tB.push_back(bench(n,rep,[&](int i){return lbr_var(s.q[i],s.c[i]);}));
    tC.push_back(bench(n,rep,[&](int i){return volfi_annulus::implied_variance_otm(s.a[i],s.c[i]);}));
    auto t0=std::chrono::steady_clock::now();
    for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_grid_batch(s.h.data(),s.c.data(),wbuf.data(),n);
    auto t1=std::chrono::steady_clock::now(); g_sink+=wbuf[0];
    tD.push_back(1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*n));
  }
  std::printf("  %-8s n=%-6d  A_volfi=%7.1f  B_lbr=%7.1f  C_scalar=%7.1f  D_gridbatch=%7.1f  (ns/eval, median)\n",
              nm,n,median(tA),median(tB),median(tC),median(tD));
}
}  // namespace

int main(int argc,char** argv){
  const int rep=env_int("BENCH_REPEATS",8);
  const int runs=env_int("BENCH_RUNS",15);
  bool dump=false; std::string dpath;
  for(int i=1;i<argc;i++){ std::string a=argv[i]; if(a=="--dump"&&i+1<argc){dump=true;dpath=argv[++i];} }
#if defined(__AVX512F__)
  const char* kIsa="AVX-512";
#elif defined(__AVX2__)&&defined(__FMA__)
  const char* kIsa="AVX2+FMA";
#else
  const char* kIsa="scalar-SIMD";
#endif
  std::printf("=== volfi-annulus Phase-7 BROAD benchmark  ISA=%s ===\n\n",kIsa);

  // ---- broad random grid: v in [0.03,8], delta (OTM) in [3e-5,0.5] -> broad h up to ~16 ----
  std::mt19937_64 gen(kSeed);
  std::uniform_real_distribution<double> dv(0.03,8.0);
  std::uniform_real_distribution<double> dld(std::log(3e-5),std::log(0.5));
  Grid G; int target=120000, tried=0;
  while((int)G.h.size()<target && tried<target*6){
    ++tried;
    double v=dv(gen); double delta=std::exp(dld(gen));
    double k=v*(0.5*v-qnorm2(delta)); double h=std::fabs(k);
    if(h>16.5||h<1e-4) continue;
    double c=volfi::black_otm_from_variance(v*v,h);
    if(!feasible(c)) continue;
    G.h.push_back(h); G.c.push_back(c); G.vt.push_back(v);
  }
  build(G);
  int N=(int)G.h.size();
  std::printf("broad random grid: N=%d feasible (from %d draws), v in [0.03,8], h up to 16.5\n\n",N,tried);

  // ---- coverage ----
  long cnt[5]={0}; long oldcov=0;
  for(int i=0;i<N;i++){ cnt[route(G.h[i],G.c[i])]++; if(old_covered(G.h[i],G.c[i])) oldcov++; }
  long newcov=cnt[R_WING]+cnt[R_LEFT]+cnt[R_CENTRAL]+cnt[R_RIGHT];
  std::printf("=== BROAD-DOMAIN COVERAGE (N=%d) ===\n",N);
  std::printf("  route counts: WING=%ld LEFT=%ld CENTRAL=%ld RIGHT=%ld EDGE=%ld\n",
              cnt[R_WING],cnt[R_LEFT],cnt[R_CENTRAL],cnt[R_RIGHT],cnt[R_EDGE]);
  std::printf("  NEW 4-chart coverage (WING+LEFT+CENTRAL+RIGHT) = %ld/%d = %.2f%%\n",newcov,N,100.0*newcov/N);
  std::printf("  OLD in-box fast path (CENTRAL table + WING only)= %ld/%d = %.2f%%   [rest was crude fallback]\n",
              oldcov,N,100.0*oldcov/N);
  std::printf("  COVERAGE GAIN: %.2f%% -> %.2f%%\n\n",100.0*oldcov/N,100.0*newcov/N);

  // ---- scalar==grid-batch bit identity over the broad grid ----
  {
    std::vector<double> wg(N);
    volfi_annulus::implied_variance_grid_batch(G.h.data(),G.c.data(),wg.data(),N);
    // Canonical invariant-(b) test: batch vs the noinline scalar entry that the
    // batch driver itself calls for the scalar fallback.  This is the fixed
    // scalar codegen the batch is *defined* to reproduce, so it is immune to the
    // -ffp-contract=fast re-fusion the compiler may apply to a separately-inlined
    // reference copy (see FINAL_REPORT: the inline-reference contraction note).
    long mism=0,gt1=0;
    FILE* mf = std::getenv("MISMATCH_DUMP") ? std::fopen("mismatch_dump.csv","w") : nullptr;
    if(mf) std::fprintf(mf,"h,c,vt,w_scalar,w_batch,route,ulp\n");
    for(int i=0;i<N;i++){ double wc=volfi_annulus::detail::scalar_fallback(G.h[i],G.c[i]);
      if(std::isnan(wc)&&std::isnan(wg[i])) continue;
      if(bd(wc)!=bd(wg[i])){ mism++; uint64_t a=bd(wc),b=bd(wg[i]); if((a>b?a-b:b-a)>1) gt1++;
        if(mf) std::fprintf(mf,"%a,%a,%a,%a,%a,%d,%llu\n",G.h[i],G.c[i],G.vt[i],wc,wg[i],
                            route(G.h[i],G.c[i]),(unsigned long long)(a>b?a-b:b-a)); } }
    if(mf) std::fclose(mf);
    std::printf("GRIDBATCH_VS_SCALAR broad grid (canonical noinline entry): mismatches=%ld (>1ulp=%ld)  [must be 0]\n\n",mism,gt1);
    if(std::getenv("DIAG_ONLY")) return 0;
  }

  // ---- accuracy vs construction variance (metric b; real mpmath rel-sigma via --dump) ----
  {
    const double EPS=2.220446049250313e-16;
    const char* lbl[4]={"WING","LEFT","CENTRAL","RIGHT"};
    std::printf("=== ACCURACY vs construction v (per chart, rel-sigma) ===\n");
    for(int rl=R_WING; rl<=R_RIGHT; ++rl){
      Grid s=subset(G,rl); if(s.h.empty()){ continue; }
      double mx=0; std::vector<double> rels;
      for(size_t i=0;i<s.h.size();++i){ double w=volfi_annulus::implied_variance_otm(s.a[i],s.c[i]);
        double v=std::sqrt(w); double rel=std::fabs(v-s.vt[i])/s.vt[i]; mx=std::max(mx,rel); rels.push_back(rel);}
      std::sort(rels.begin(),rels.end()); double med=rels[rels.size()/2];
      std::printf("  %-8s n=%-6zu  max=%.3e (%.2f ulp)  median=%.3e (%.2f ulp)\n",
                  lbl[rl],s.h.size(),mx,mx/EPS,med,med/EPS);
    }
    std::printf("\n");
  }

  if(dump){
    FILE* f=std::fopen(dpath.c_str(),"w");
    std::fprintf(f,"idx,h_hex,c_hex,vt_hex,wc_hex,route\n");
    for(int i=0;i<N;i++){ double wc=volfi_annulus::implied_variance_otm(G.a[i],G.c[i]);
      std::fprintf(f,"%d,%a,%a,%a,%a,%d\n",i,G.h[i],G.c[i],G.vt[i],wc,route(G.h[i],G.c[i])); }
    std::fclose(f); std::printf("DUMP wrote %d rows to %s\n\n",N,dpath.c_str());
  }

  std::printf("=== PER-CHART TIMING (BENCH_RUNS=%d REPEATS=%d), mixed broad grid subset per chart ===\n",runs,rep);
  time_chart("CENTRAL",subset(G,R_CENTRAL),runs,rep);
  time_chart("LEFT",   subset(G,R_LEFT),   runs,rep);
  time_chart("RIGHT",  subset(G,R_RIGHT),  runs,rep);
  time_chart("WING",   subset(G,R_WING),   runs,rep);
  std::printf("\n");

  // ---- full mixed broad-grid batch headline ----
  {
    std::vector<double> tC,tD; std::vector<double> wbuf(N);
    std::vector<double> tBall;
    for(int r=0;r<runs;r++){
      tC.push_back(bench(N,rep,[&](int i){return volfi_annulus::implied_variance_otm(G.a[i],G.c[i]);}));
      tBall.push_back(bench(N,rep,[&](int i){return lbr_var(G.q[i],G.c[i]);}));
      auto t0=std::chrono::steady_clock::now();
      for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_grid_batch(G.h.data(),G.c.data(),wbuf.data(),N);
      auto t1=std::chrono::steady_clock::now(); g_sink+=wbuf[0];
      tD.push_back(1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*N));
    }
    std::printf("=== FULL broad mixed grid (all charts) ===\n");
    rpt("mixed","B_lbr_all",tBall); rpt("mixed","C_scalar_all",tC); rpt("mixed","D_gridbatch_all",tD);
    std::printf("  batch speedup vs scalar = %.2fx (blended; CENTRAL subset vectorized, endpoints scalar)\n\n",median(tC)/median(tD));
  }

  // ---- fixed-h SURFACE per chart (same-cell throughput ceiling) ----
  //  CENTRAL: h=1.0 v in [0.05,2].  LEFT: h=0.2 v in [0.3,1.6].  RIGHT: h=1.0 v in [2.1,8].
  auto surface=[&](const char* nm,double h,double vlo,double vhi){
    volfi_annulus::context aq(h); volfi::otm_context vq(h);
    std::vector<double> cs; int M=4096;
    for(int i=0;i<M;i++){ double v=vlo+(vhi-vlo)*(i+0.5)/M; double c=volfi::black_otm_from_variance(v*v,h);
      if(feasible(c)) cs.push_back(c); }
    std::sort(cs.begin(),cs.end());
    int n=(int)cs.size(); std::vector<double> w(n);
    std::vector<double> tCs,tDs,tAs,tBs;
    for(int r=0;r<runs;r++){
      auto t0=std::chrono::steady_clock::now();
      for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_otm_batch(aq,cs.data(),w.data(),n);
      auto t1=std::chrono::steady_clock::now(); g_sink+=w[0];
      tDs.push_back(1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*n));
      tCs.push_back(bench(n,rep,[&](int i){return volfi_annulus::implied_variance_otm(aq,cs[i]);}));
      tAs.push_back(bench(n,rep,[&](int i){return volfi::implied_variance_otm(vq,cs[i]);}));
      tBs.push_back(bench(n,rep,[&](int i){return lbr_var(vq,cs[i]);}));
    }
    std::printf("  %-8s h=%.2f v in [%.2f,%.2f] n=%-5d  A_volfi=%7.1f  B_lbr=%7.1f  C_scalar=%7.1f  D_batch=%7.1f\n",
                nm,h,vlo,vhi,n,median(tAs),median(tBs),median(tCs),median(tDs));
  };
  std::printf("=== FIXED-h SURFACE per chart (ns/eval median; Phase-8 D_batch vectorizes CENTRAL+LEFT+RIGHT) ===\n");
  std::printf("  -- CENTRAL-clean surfaces (v above wing seam h/sqrt6, below v=2; chart-pure) --\n");
  surface("CENTRAL",1.0,0.45,1.95);
  surface("CENTRAL",2.0,0.85,1.95);
  surface("CENTRALc",1.0,0.05,2.0);   // legacy wing-contaminated range, kept for reference
  std::printf("  -- LEFT-heavy surfaces (h<H_ATM_HI, v<=1.70; Phase-8 batch runs LEFT SIMD kernel) --\n");
  surface("LEFT",0.10,0.20,1.60);
  surface("LEFT",0.20,0.30,1.60);
  surface("LEFT",0.28,0.30,1.60);
  std::printf("  -- RIGHT-heavy surfaces (v>2; Phase-8 batch runs RIGHT SIMD kernel) --\n");
  surface("RIGHT",0.50,2.10,8.0);
  surface("RIGHT",1.00,2.10,8.0);
  surface("RIGHT",3.00,2.10,8.0);
  std::printf("\n");

  // ---- MARKET-REALISTIC FEED (2024 SPX EOD tradeable, ACTUAL prices) ----------
  // Loads market_feed.csv ("h c" per line; h=|log(F/K)|, c=undiscounted OTM price/F),
  // a 30k uniform resample of the 2024 SPX TRADEABLE book (uncrossed two-sided,
  // mid>=$0.50, spread/mid<=25%, OI>=100 & vol>=10, tau>=7d; VENDOR forwards matched
  // on settlement; every quote parity-projected to the OTM-call branch).  The c are
  // the ACTUAL projected quote prices -- tick rounding, bid/ask noise and parity
  // projection included -- NOT prices regenerated from vendor IV, so the route mix
  // (~90% LEFT / ~5% WING / ~5% CENTRAL, wing ~98% downside puts) is the live one.
  // TIMING CONVENTION (all-inside): every method starts from the raw (h,c) pair
  // inside the timed loop -- LBR pays its beta = c*exp(-h/2) input transform and
  // our scalar entry builds its per-h context; nothing is precomputed for anybody.
  {
    Grid M; std::ifstream fin("market_feed.csv");
    if(!fin){ std::printf("=== MARKET-REALISTIC FEED : market_feed.csv not found (skipped) ===\n\n"); }
    else{
      std::string ln;
      while(std::getline(fin,ln)){
        if(ln.empty()||ln[0]=='#') continue;
        std::istringstream is(ln); double h,c; if(!(is>>h>>c)) continue;
        if(!(h>1e-4&&h<16.5)||!feasible(c)) continue;
        M.h.push_back(h); M.c.push_back(c); M.vt.push_back(0.0);
      }
      int NM=(int)M.h.size();
      long mc[5]={0}; for(int i=0;i<NM;i++) mc[route(M.h[i],M.c[i])]++;
      std::printf("=== MARKET-REALISTIC FEED (2024 SPX tradeable, actual prices, N=%d) ===\n",NM);
      { // bit-identity ON THIS FEED: a LEFT-heavy feed dispatches to the SPECULATIVE
        // driver, which the broad-grid DIAG (two-pass dispatch) does not exercise.
        std::vector<double> wg(NM); long mism=0;
        volfi_annulus::implied_variance_grid_batch(M.h.data(),M.c.data(),wg.data(),NM);
        for(int i=0;i<NM;i++){ double ws=volfi_annulus::detail::scalar_fallback(M.h[i],M.c[i]);
          if(std::isnan(ws)&&std::isnan(wg[i])) continue;
          if(bd(ws)!=bd(wg[i])) ++mism; }
        std::printf("  GRIDBATCH_VS_SCALAR market feed (speculative driver): mismatches=%ld  [must be 0]\n",mism);
      }
      std::printf("  route mix: WING=%.2f%% LEFT=%.2f%% CENTRAL=%.2f%% RIGHT=%.2f%% EDGE=%.2f%%\n",
                  100.0*mc[R_WING]/NM,100.0*mc[R_LEFT]/NM,100.0*mc[R_CENTRAL]/NM,
                  100.0*mc[R_RIGHT]/NM,100.0*mc[R_EDGE]/NM);
      std::vector<double> tBm,tCm,tDm; std::vector<double> wbuf(NM);
      for(int r=0;r<runs;r++){
        tBm.push_back(bench(NM,rep,[&](int i){
          double beta=M.c[i]*std::exp(-0.5*M.h[i]);            // LBR's input transform, in the clock
          double s=NormalisedImpliedBlackVolatility(beta,-M.h[i],1.0); return s*s;}));
        tCm.push_back(bench(NM,rep,[&](int i){
          return volfi_annulus::implied_variance_otm(M.h[i],M.c[i]);}));  // context built inside
        auto t0=std::chrono::steady_clock::now();
        for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_grid_batch(M.h.data(),M.c.data(),wbuf.data(),NM);
        auto t1=std::chrono::steady_clock::now(); g_sink+=wbuf[0];
        tDm.push_back(1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*NM));
      }
      rpt("market","B_lbr_feed",tBm); rpt("market","C_scalar_feed",tCm); rpt("market","D_gridbatch_feed",tDm);
      std::printf("  batch speedup vs scalar = %.2fx ; vs LBR scalar = %.2fx\n\n",
                  median(tCm)/median(tDm), median(tBm)/median(tDm));

      // ---- STREAMING re-inversion on the SAME market feed ----------------------
      // Steady state: each node re-priced from its previous snapshot.  Seed w_prev
      // with the cold result stale by a 0.5% vol move, then warm-refine (no routing,
      // no drain).  Reports the fast 2-step (live streaming) and 3-step warm batch,
      // both on the identical feed and host as the cold row above.
      {
        std::vector<double> wcold(NM); std::vector<double> Wprev(NM);
        volfi_annulus::implied_variance_grid_batch(M.h.data(),M.c.data(),wcold.data(),NM);
        for(int i=0;i<NM;i++){ double v=std::sqrt(wcold[i])*1.005; Wprev[i]=v*v; }
        std::vector<double> tW2,tW3;
        for(int r=0;r<runs;r++){
          auto t0=std::chrono::steady_clock::now();
          for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_warm_batch(M.h.data(),M.c.data(),Wprev.data(),wbuf.data(),NM,2);
          auto t1=std::chrono::steady_clock::now(); g_sink+=wbuf[0];
          tW2.push_back(1e9*std::chrono::duration<double>(t1-t0).count()/((double)rep*NM));
          auto t2=std::chrono::steady_clock::now();
          for(int rr=0;rr<rep;rr++) volfi_annulus::implied_variance_warm_batch(M.h.data(),M.c.data(),Wprev.data(),wbuf.data(),NM,3);
          auto t3=std::chrono::steady_clock::now(); g_sink+=wbuf[0];
          tW3.push_back(1e9*std::chrono::duration<double>(t3-t2).count()/((double)rep*NM));
        }
        rpt("stream","E_warm2_feed",tW2); rpt("stream","E_warm3_feed",tW3);
        std::printf("  warm-2step vs cold gridbatch = %.2fx faster ; vs LBR = %.2fx\n\n",
                    median(tDm)/median(tW2), median(tBm)/median(tW2));
      }
    }
  }

  if(g_sink==123456789.0) std::printf("%g\n",(double)g_sink);
  std::printf("done.\n");
  return 0;
}
