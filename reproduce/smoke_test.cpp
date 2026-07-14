// smoke_test.cpp -- self-contained broad-domain round-trip smoke test for the
// Phase-8 volfi-annulus inverter (umbrella header only).  Phase-8 vectorizes the
// LEFT and RIGHT charts in the batch drivers; this smoke exercises all four charts
// + corners/seams and checks scalar==grid-batch bit-for-bit.  It is NOT the
// machine-precision gate (that is verify_vec.py/.cpp vs mpmath); it checks that the
// whole broad domain -- including the h=0.3/v=2 corner, v>2, the chart seams,
// deep-OTM and large h -- inverts back to the generating vol at 1e-9 round-trip.
//
// Build (AVX-512): g++ -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math smoke_test.cpp -o smoke
// Build (AVX2)   : add -mno-avx512f
#include "volfi_annulus_all.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
using namespace volfi_annulus;

static const char* rlabel(double h,double c){
  if(!(c>0.0)||c>=1.0) return "EDGE";
  if(c<br::cwing_price(h)) return "WING";
  if(h<H_ATM_HI){ double ctl=1.0-br::onem_otm(h,volfi_annulus_broadrange::LEFT_RIGHT_VSEAM,std::exp(h));
    return (c<=ctl)?"LEFT":"RIGHT"; }
  if(h<=H_BOX && c<=br::c2_price(h)) return "CENTRAL";
  return "RIGHT";
}
static inline uint64_t B(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }

int main(){
  // (h,v) probes across every chart + the flagged corners/seams.
  std::vector<std::pair<double,double>> pts;
  auto add=[&](double h,double v){ pts.push_back({h,v}); };
  // CENTRAL box
  for(double h : {0.35,0.6,1.0,2.0,4.0,6.0}) for(double v : {0.1,0.3,0.6,1.0,1.5,1.9}) add(h,v);
  // LEFT small-h
  for(double h : {0.005,0.02,0.08,0.15,0.25,0.299}) for(double v : {0.05,0.2,0.6,1.0,1.4,1.65}) add(h,v);
  // the former FAILING corner h->0.3, v->2
  for(double h : {0.26,0.28,0.294,0.299}) for(double v : {1.80,1.90,1.96,1.99}) add(h,v);
  // LEFT<->RIGHT seam v=1.70 both sides, h<0.3
  for(double h : {0.05,0.15,0.28}) for(double v : {1.66,1.69,1.71,1.74}) add(h,v);
  // RIGHT v>2 (any h)
  for(double h : {0.05,0.3,1.0,3.0,6.6}) for(double v : {2.1,3.0,4.5,6.0,7.5}) add(h,v);
  // v=2 seam both sides at h in box
  for(double h : {0.5,1.5,4.0}) for(double v : {1.95,1.99,2.01,2.05}) add(h,v);
  // deep-OTM corner (small h, large W just under 3)
  for(double h : {0.01,0.1,0.29}) for(double W : {1.6,2.2,2.8,2.95}) add(h,h/std::sqrt(2*W));
  // large h up to 16 (WING) -- the other former failing region
  for(double h : {8.0,11.0,14.0,14.84,16.0}) for(double W : {4.0,20.0,120.0,150.0,155.0,250.0}) add(h,h/std::sqrt(2*W));

  int n=(int)pts.size();
  std::vector<double> hs(n),cs(n),vt(n);
  int used=0;
  for(int i=0;i<n;i++){
    double h=pts[i].first, v=pts[i].second, w=v*v;
    double c=volfi::black_otm_from_variance(w,h);
    if(!(c>0.0&&c<1.0&&1.0-c>1e-16)) continue;         // skip float64-infeasible
    hs[used]=h; cs[used]=c; vt[used]=v; ++used;
  }
  n=used; hs.resize(n); cs.resize(n); vt.resize(n);

  // scalar invert + round-trip
  double worst=0.0; const char* wroute=""; double wh=0,wv=0; int fail=0;
  std::vector<double> ws(n);
  for(int i=0;i<n;i++){
    double w=implied_variance_otm(hs[i],cs[i]); ws[i]=w;
    double vr=std::sqrt(w);
    double rel=std::fabs(vr-vt[i])/vt[i];
    if(!std::isfinite(w)||w<=0.0){ ++fail; std::printf("NONFINITE h=%.4g c=%.6g\n",hs[i],cs[i]); }
    if(rel>worst){ worst=rel; wroute=rlabel(hs[i],cs[i]); wh=hs[i]; wv=vt[i]; }
  }
  // batch (grid) bit-identity
  std::vector<double> wg(n);
  implied_variance_grid_batch(hs.data(),cs.data(),wg.data(),n);
  long mism=0;
  for(int i=0;i<n;i++) if(B(ws[i])!=B(wg[i])) ++mism;

  std::printf("smoke: %d broad probes across WING/LEFT/CENTRAL/RIGHT + corners/seams\n",n);
  std::printf("  worst round-trip rel-vol = %.3e  (route=%s, h=%.4g, v=%.4g)\n",worst,wroute,wh,wv);
  std::printf("  nonfinite/nonpositive    = %d\n",fail);
  std::printf("  scalar==grid-batch mismatches = %ld\n",mism);
  // tolerance is loose (1e-9): the forward pricer here is a plain float64 Black
  // price that cancels for deep-OTM; the machine-precision gate lives in verify_broad.
  bool pass = (fail==0) && (mism==0) && (worst < 1e-9);
  std::printf("%s\n", pass ? "SMOKE PASS" : "SMOKE FAIL");
  return pass?0:1;
}
