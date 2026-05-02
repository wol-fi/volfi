#include <algorithm>
#include <cmath>
#include <cstdio>
#include <volfi/volfi.hpp>

constexpr int ncase=164;
constexpr double sr2pi=2.50662827463100050241576528481104525;

inline double qnorm0(double p){
 const double a1=-3.969683028665376e+01,a2=2.209460984245205e+02,a3=-2.759285104469687e+02,a4=1.383577518672690e+02,a5=-3.066479806614716e+01,a6=2.506628277459239e+00;
 const double b1=-5.447609879822406e+01,b2=1.615858368580409e+02,b3=-1.556989798598866e+02,b4=6.680131188771972e+01,b5=-1.328068155288572e+01;
 const double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,c3=-2.400758277161838e+00,c4=-2.549732539343734e+00,c5=4.374664141464968e+00,c6=2.938163982698783e+00;
 const double d1=7.784695709041462e-03,d2=3.224671290700398e-01,d3=2.445134137142996e+00,d4=3.754408661907416e+00;
 double q,r;
 if(p<0.02425){q=std::sqrt(-2*std::log(p)); return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);}
 if(p<=0.97575){q=p-0.5; r=q*q; return ((((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q)/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);}
 q=std::sqrt(-2*std::log1p(-p)); return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);
}

inline double qnorm2(double p){
 double x=qnorm0(p);
 for(int i=0;i<2;i++){double e=volfi::phi_cdf(x)-p; double u=e*sr2pi*std::exp(0.5*x*x); x-=u/(1+0.5*x*u);}
 return x;
}

int main(){
 double delta[4]={0.55,0.70,0.80,0.95};
 double mav=0,mxv=0;
 int ix=0;
 for(int iv=0;iv<41;iv++){
  double v=(iv==0)?0.01:0.05*iv;
  for(int id=0;id<4;id++){
   double k=v*(0.5*v-qnorm2(delta[id]));
   double h=std::fabs(k), w=v*v;
   volfi::otm_context q(h);
   double c=volfi::black_otm_from_variance(w,q);
   double ve=std::sqrt(volfi::implied_variance_otm(q,c));
   double e=std::fabs(ve-v);
   mav+=e; mxv=std::max(mxv,e); ix++;
  }
 }
 mav/=ix;
 std::printf("cases %d\nmean_abs_vol %.17g\nmax_abs_vol %.17g\n",ix,mav,mxv);
 return mxv<1e-14 ? 0 : 1;
}
