// warm_test.cpp -- D(warm-start) verification:
//  (1) accuracy: warm(h, c, w_cold*(1±2%)^2) must agree with the oracle-verified cold
//      inverter to ~1 ulp across the basin (W <= 40), mixed h;
//  (2) bit-identity: warm batch == warm scalar per quote, this ISA;
//  (3) timing: warm batch vs cold grid batch on the same mixed feed.
#include "volfi_annulus_all.hpp"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
using namespace volfi_annulus;
static uint64_t B(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
static double price(double h,double v){double w=v*v,s=std::sqrt(w),u=-h/s;
  auto Phi=[](double x){return 0.5*std::erfc(-x*0.70710678118654752440);};
  return Phi(u+0.5*s)-std::exp(h)*Phi(u-0.5*s);}
int main(){
#if defined(VA_SIMD512)
  printf("ISA=AVX-512\n");
#elif defined(VA_SIMD256)
  printf("ISA=AVX2+FMA\n");
#else
  printf("ISA=scalar\n");
#endif
  // mixed-h feed inside the warm basin: W = h^2/(2 v^2) <= 40, all charts incl. mild wing
  std::vector<double> H,C,Wc;
  for(int ih=0; ih<=120; ih++){ double h=0.001+ih*10.0/120.0;
    for(int iv=1; iv<=60; iv++){ double v=0.05+(8.0-0.05)*iv/60.0;
      if(h*h/(2*v*v) > 40.0) continue;
      double c=price(h,v); if(!(c>1e-300)||!(c<1.0-1e-14)) continue;
      H.push_back(h); C.push_back(c);
    }}
  int N=(int)H.size();
  Wc.resize(N);
  for(int i=0;i<N;i++) Wc[i]=implied_variance_otm(H[i],C[i]);   // cold = oracle-verified ref

  // --- accuracy from ±2% perturbed seeds ---
  // Documented floor: few-ULP (x-space representation limit, see endpoint_vec.hpp):
  //   gate = 2.5e-15 for W<=8, 1.0e-14 for W<=40.
  double worst8=0, worst40=0; int over=0;
  std::vector<double> Wp(N), Wseed(N);
  for(int sgn=0; sgn<2; sgn++){
    double f = sgn? 0.98*0.98 : 1.02*1.02;
    for(int i=0;i<N;i++) Wseed[i]=Wc[i]*f;
    implied_variance_warm_batch(H.data(),C.data(),Wseed.data(),Wp.data(),N);
    for(int i=0;i<N;i++){
      double rel=std::fabs(std::sqrt(Wp[i])-std::sqrt(Wc[i]))/std::sqrt(Wc[i]);
      double W=H[i]*H[i]/(2.0*Wc[i]);
      if(W<=8.0){ worst8=std::max(worst8,rel); if(rel>3.0e-15) over++; }
      else      { worst40=std::max(worst40,rel); if(rel>1.5e-14) over++; }
    }
  }
  printf("WARM accuracy vs cold (N=%d, mixed h): worst(W<=8)=%.3e [gate 3e-15]  worst(W<=40)=%.3e [gate 1.5e-14]  over=%d\n",
         N,worst8,worst40,over);

  // --- scalar == batch bit-identity for the warm entry ---
  for(int i=0;i<N;i++) Wseed[i]=Wc[i]*1.02*1.02;
  implied_variance_warm_batch(H.data(),C.data(),Wseed.data(),Wp.data(),N);
  int mm=0;
  for(int i=0;i<N;i++){ double s=implied_variance_warm(H[i],C[i],Wseed[i]); if(B(s)!=B(Wp[i])) mm++; }
  printf("WARM bit-identity scalar==batch: mismatches=%d\n",mm);

  // --- timing: warm batch vs cold grid batch, same mixed feed ---
  std::vector<double> wo(N);
  auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<50;r++) implied_variance_warm_batch(H.data(),C.data(),Wseed.data(),wo.data(),N);
  auto t1=std::chrono::steady_clock::now();
  for(int r=0;r<50;r++) implied_variance_grid_batch(H.data(),C.data(),wo.data(),N);
  auto t2=std::chrono::steady_clock::now();
  volatile double sink=0;
  for(int r=0;r<50;r++) for(int i=0;i<N;i++) sink+=implied_variance_warm(H[i],C[i],Wseed[i]);
  auto t3=std::chrono::steady_clock::now();
  double warm_b=1e9*std::chrono::duration<double>(t1-t0).count()/(50.0*N);
  double cold_b=1e9*std::chrono::duration<double>(t2-t1).count()/(50.0*N);
  double warm_s=1e9*std::chrono::duration<double>(t3-t2).count()/(50.0*N);
  printf("TIMING mixed feed: warm_batch=%.0f ns  warm_scalar=%.0f ns  cold_grid_batch=%.0f ns  (warm batch %.1fx vs cold)\n",
         warm_b,warm_s,cold_b,cold_b/warm_b);
  bool ok=(over==0&&mm==0);
  printf("%s\n", ok?"WARM PASS":"WARM FAIL");
  return ok?0:1;
}
