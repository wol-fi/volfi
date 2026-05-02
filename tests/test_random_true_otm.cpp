#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <volfi/volfi.hpp>
int main(){
 const int n=200000; std::mt19937_64 rng(20260502ULL);
 std::uniform_real_distribution<double> ud(0.01,0.9), uv(0.01,2.0);
 double mae=0,mxe=0,mre=0; int nbad=0;
 for(int i=0;i<n;i++){
  double d=ud(rng), v=uv(rng), h=std::abs(v*(0.5*v-volfi::qnorm(d))), w=v*v;
  volfi::otm_context q(h);
  double c=volfi::black_otm_from_variance(w,q);
  double vv=std::sqrt(volfi::implied_variance_otm(q,c));
  double e=std::abs(vv-v);
  mae+=e; mxe=std::max(mxe,e); mre=std::max(mre,e/v); if(e>1e-14) nbad++;
 }
 std::printf("random_true_otm cases %d\ndelta_distribution uniform_0.01_0.9\nv_distribution uniform_0.01_2.0\nmean_abs_vol %.17g\nmax_abs_vol %.17g\nmax_rel_vol %.17g\nn_abs_gt_1e-14 %d\n",n,mae/n,mxe,mre,nbad);
 return mxe<1e-14 && nbad==0 ? 0 : 1;
}
