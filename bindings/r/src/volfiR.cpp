#include <Rcpp.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

static double ncdf(double x){return 0.5*std::erfc(-x/std::sqrt(2.0));}

static double qnorm0(double p){
 const double a1=-3.969683028665376e+01,a2=2.209460984245205e+02,a3=-2.759285104469687e+02,a4=1.383577518672690e+02,a5=-3.066479806614716e+01,a6=2.506628277459239e+00;
 const double b1=-5.447609879822406e+01,b2=1.615858368580409e+02,b3=-1.556989798598866e+02,b4=6.680131188771972e+01,b5=-1.328068155288572e+01;
 const double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,c3=-2.400758277161838e+00,c4=-2.549732539343734e+00,c5=4.374664141464968e+00,c6=2.938163982698783e+00;
 const double d1=7.784695709041462e-03,d2=3.224671290700398e-01,d3=2.445134137142996e+00,d4=3.754408661907416e+00;
 double q,r;
 if(p<0.02425){q=std::sqrt(-2*std::log(p)); return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);}
 if(p<=0.97575){q=p-0.5; r=q*q; return ((((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q)/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);}
 q=std::sqrt(-2*std::log1p(-p)); return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);
}

static double qnorm(double p){
 double x=qnorm0(p);
 const double sr2pi=2.5066282746310005024;
 for(int i=0;i<2;i++){double e=ncdf(x)-p; double u=e*sr2pi*std::exp(0.5*x*x); x-=u/(1+0.5*x*u);}
 return x;
}

static double black_otm0(double w,double h){
 double s=std::sqrt(w), u=-h/s;
 return ncdf(u+0.5*s)-std::exp(h)*ncdf(u-0.5*s);
}

static double iv_otm0(double h,double c){
 if(!(h>=0) || !(c>0) || !(c<1)) return NA_REAL;
 double lo=1e-16, hi=1.0;
 while(black_otm0(hi,h)<c && hi<4096.0) hi*=2.0;
 for(int i=0;i<80;i++){
  double mid=0.5*(lo+hi);
  if(black_otm0(mid,h)<c) lo=mid; else hi=mid;
 }
 return 0.5*(lo+hi);
}

// [[Rcpp::export]]
NumericVector iv(NumericVector k, NumericVector c){
 int n=std::max(k.size(),c.size());
 NumericVector out(n);
 for(int i=0;i<n;i++){
  double x=k[i%k.size()], y=c[i%c.size()];
  if(x==0){double z=2*qnorm(0.5*(1+y)); out[i]=z*z;}
  else if(x>0) out[i]=iv_otm0(x,y);
  else out[i]=iv_otm0(-x,1+std::exp(-x)*(y-1));
 }
 return out;
}

// [[Rcpp::export]]
NumericVector iv_atm(NumericVector c){
 int n=c.size();
 NumericVector out(n);
 for(int i=0;i<n;i++){double z=2*qnorm(0.5*(1+c[i])); out[i]=z*z;}
 return out;
}

// [[Rcpp::export]]
NumericVector black(NumericVector w, NumericVector h){
 int n=std::max(w.size(),h.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=black_otm0(w[i%w.size()],std::abs(h[i%h.size()]));
 return out;
}
