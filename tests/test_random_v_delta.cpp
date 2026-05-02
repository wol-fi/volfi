#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <volfi/volfi.hpp>

int main(){
 const int n=200000;
 const unsigned long long seed=20260502ULL;
 std::mt19937_64 rng(seed);
 std::uniform_real_distribution<double> ud(0.5,0.99), uv(0.01,2.0);
 double mae=0,mxe=0,mre=0;
 int ngt=0;
 for(int i=0;i<n;i++){
  double delta=ud(rng);
  double v=uv(rng);
  double k=v*(0.5*v-volfi::qnorm(delta));
  double h=std::abs(k);
  volfi::otm_context q(h);
  double c=volfi::black_otm_from_variance(v*v,q);
  double vv=std::sqrt(volfi::implied_variance_otm(q,c));
  double e=std::abs(vv-v);
  mae+=e;
  mxe=std::max(mxe,e);
  mre=std::max(mre,e/v);
  if(e>1e-14) ngt++;
 }
 mae/=n;
 std::printf("random_v_delta_seed %llu\n",seed);
 std::printf("cases %d\n",n);
 std::printf("delta_distribution uniform_0.5_0.99\n");
 std::printf("v_distribution uniform_0.01_2.0\n");
 std::printf("mean_abs_vol %.17g\n",mae);
 std::printf("max_abs_vol %.17g\n",mxe);
 std::printf("max_rel_vol %.17g\n",mre);
 std::printf("n_abs_gt_1e-14 %d\n",ngt);
 return (mxe<1e-14 && ngt==0) ? 0 : 1;
}
