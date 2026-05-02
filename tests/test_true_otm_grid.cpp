#include <algorithm>
#include <cmath>
#include <cstdio>
#include <volfi/volfi.hpp>
int main(){
 double delta[4]={0.05,0.20,0.30,0.45};
 double mae=0,mxe=0,mre=0; int n=0,nbad=0;
 for(int iv=0;iv<41;iv++){
  double v=(iv==0)?0.01:0.05*iv;
  for(double d:delta){
   double h=std::abs(v*(0.5*v-volfi::qnorm(d))), w=v*v;
   volfi::otm_context q(h);
   double c=volfi::black_otm_from_variance(w,q);
   double vv=std::sqrt(volfi::implied_variance_otm(q,c));
   double e=std::abs(vv-v);
   mae+=e; mxe=std::max(mxe,e); mre=std::max(mre,e/v); if(e>1e-14) nbad++; n++;
  }
 }
 std::printf("true_otm_grid cases %d\nmean_abs_vol %.17g\nmax_abs_vol %.17g\nmax_rel_vol %.17g\nn_abs_gt_1e-14 %d\n",n,mae/n,mxe,mre,nbad);
 return mxe<1e-14 && nbad==0 ? 0 : 1;
}
