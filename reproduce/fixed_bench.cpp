// Lean fixed-h surface: bit-identity (scalar==batch) + scalar-vs-AVX2 speedup, LEFT & RIGHT & CENTRAL.
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
static void run(double h,double vlo,double vhi,const char* tag){
  int M=4096; static std::vector<double> cc,wo; cc.resize(M);wo.resize(M);
  context q(h);
  for(int i=0;i<M;i++){double v=vlo+(vhi-vlo)*i/M; cc[i]=price(h,v);}
  implied_variance_otm_batch(q,cc.data(),wo.data(),M);
  int mm=0; for(int i=0;i<M;i++){double s=implied_variance_otm(q,cc[i]); if(B(s)!=B(wo[i]))mm++;}
  // route census (confirm the surface is chart-clean)
  int nC=0,nL=0,nR=0,nW=0,nE=0;
  for(int i=0;i<M;i++){double c=cc[i];
    if(!(c>0.0)||c>=1.0){++nE;continue;}
    if(c<br::cwing_price(h)){++nW;continue;}
    int band; if(detail::grid_central_cell(h,c,detail::bits_of(c),band)>=0){++nC;continue;}
    int rt=detail::grid_endpoint_route(h,c); if(rt==1)++nL; else if(rt==2)++nR; else ++nE; }
  auto t0=std::chrono::steady_clock::now(); volatile double sink=0;
  for(int r=0;r<300;r++) for(int i=0;i<M;i++) sink+=implied_variance_otm(q,cc[i]);
  auto t1=std::chrono::steady_clock::now();
  for(int r=0;r<300;r++) implied_variance_otm_batch(q,cc.data(),wo.data(),M);
  auto t2=std::chrono::steady_clock::now();
  double sc=1e9*std::chrono::duration<double>(t1-t0).count()/(300.0*M);
  double ba=1e9*std::chrono::duration<double>(t2-t1).count()/(300.0*M);
  printf("%s h=%.2f [C%d L%d R%d W%d E%d]: mismatches=%d  scalar=%.0f ns  batch=%.0f ns  speedup=%.1fx\n",
         tag,h,nC,nL,nR,nW,nE,mm,sc,ba,sc/ba);
}
int main(){
#if defined(VA_SIMD512)
  printf("ISA=AVX-512\n");
#elif defined(VA_SIMD256)
  printf("ISA=AVX2+FMA\n");
#else
  printf("ISA=scalar\n");
#endif
  run(0.20,0.45,1.55,"LEFT   ");
  run(1.00,2.20,7.5, "RIGHT  ");
  run(1.00,0.80,1.75,"CENTRAL");
  return 0;
}
