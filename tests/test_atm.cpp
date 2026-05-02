#include <algorithm>
#include <cmath>
#include <cstdio>
#include <volfi/volfi.hpp>
int main(){double mx=0,mean=0;int n=0;for(int iv=0;iv<41;iv++){double v=iv?0.05*iv:0.01;double c=2*volfi::phi_cdf(0.5*v)-1;double w=volfi::implied_variance_call_normalised(0,c);double e=std::fabs(std::sqrt(w)-v);mean+=e;mx=std::max(mx,e);n++;}mean/=n;std::printf("cases %d\nmean_abs_vol %.17g\nmax_abs_vol %.17g\n",n,mean,mx);return mx<1e-14?0:1;}
