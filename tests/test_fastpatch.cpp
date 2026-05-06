#include <volfi/volfi.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main(){
 std::vector<double> e;
 auto add=[&](double h,double c){
  volfi::otm_context q(h);
  double w0=volfi::seed_otm(q,c), wf=0.0;
  if(!volfi::fastpatch_variance(w0,q,c,wf)) return 1;
  double wh=volfi::halley_variance(w0,q,c);
  e.push_back(std::fabs(std::sqrt(wf)-std::sqrt(wh)));
  return 0;
 };
 for(int ih=0;ih<20;ih++) for(int ic=0;ic<20;ic++){
  double h0=0.05+0.0175*ih,c0=0.20+0.0175*ic;
  for(int a=0;a<=8;a++) for(int b=0;b<=9;b++){
   double h=std::min(0.40,h0+0.0175*a/8.0);
   double c=std::min(0.55,c0+0.0175*b/9.0);
   if(add(h,c)) return 2;
  }
 }
 std::mt19937_64 rng(20260506);
 std::uniform_real_distribution<double> uh(0.05,0.40),uc(0.20,0.55);
 for(int i=0;i<100000;i++) if(add(uh(rng),uc(rng))) return 3;
 std::sort(e.begin(),e.end());
 long double s=0; for(double x:e) s+=x;
 double mx=e.back(),q999=e[(size_t)(0.999*(e.size()-1))],med=e[e.size()/2],mean=(double)(s/e.size());
 std::printf("fastpatch cases %zu\n",e.size());
 std::printf("mean_abs_vol_vs_halley %.17g\n",mean);
 std::printf("median_abs_vol_vs_halley %.17g\n",med);
 std::printf("q999_abs_vol_vs_halley %.17g\n",q999);
 std::printf("max_abs_vol_vs_halley %.17g\n",mx);
 std::printf("n_abs_gt_3e-15 %zu\n",std::count_if(e.begin(),e.end(),[](double x){return x>3e-15;}));
 return mx<=3e-15?0:1;
}
