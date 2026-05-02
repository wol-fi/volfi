#include <volfi/volfi.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

constexpr int ncase=164;
constexpr int reps=5000;
constexpr int runs=9;
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

inline double now(){timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec;}

void stat(const char* name,std::vector<double> x){
 std::sort(x.begin(),x.end());
 double s=0;
 for(double y:x) s+=y;
 std::printf("%s_mean_ns_per_eval %.17g\n",name,s/x.size());
 std::printf("%s_median_ns_per_eval %.17g\n",name,x[x.size()/2]);
 std::printf("%s_min_ns_per_eval %.17g\n",name,x.front());
 std::printf("%s_max_ns_per_eval %.17g\n",name,x.back());
}

template<class F> double bench(F f){
 volatile double sink=0;
 double t0=now();
 for(int r=0;r<reps;r++) for(int i=0;i<ncase;i++) sink+=std::sqrt(f(i));
 double t1=now();
 if(sink==123456789) std::printf("%g\n",(double)sink);
 return 1e9*(t1-t0)/(reps*ncase);
}

template<class F> void acc(const char* name,F f,const double* wt,const double* vt){
 double maw=0,mxw=0,mrw=0,mav=0,mxv=0,mrv=0;
 for(int i=0;i<ncase;i++){
  double wh=f(i),v=std::sqrt(wh),ew=std::fabs(wh-wt[i]),ev=std::fabs(v-vt[i]);
  maw+=ew; mxw=std::max(mxw,ew); mrw=std::max(mrw,ew/wt[i]);
  mav+=ev; mxv=std::max(mxv,ev); mrv=std::max(mrv,ev/vt[i]);
 }
 std::printf("%s_mean_abs_variance_error %.17g\n",name,maw/ncase);
 std::printf("%s_max_abs_variance_error %.17g\n",name,mxw);
 std::printf("%s_max_rel_variance_error %.17g\n",name,mrw);
 std::printf("%s_mean_abs_vol_error %.17g\n",name,mav/ncase);
 std::printf("%s_max_abs_vol_error %.17g\n",name,mxv);
 std::printf("%s_max_rel_vol_error %.17g\n",name,mrv);
}

int main(){
 double delta[4]={0.55,0.70,0.80,0.95};
 double c[ncase],wt[ncase],vt[ncase];
 std::vector<volfi::otm_context> q;
 q.reserve(ncase);
 int ix=0;
 for(int iv=0;iv<41;iv++){
  double v=(iv==0)?0.01:0.05*iv;
  for(int id=0;id<4;id++){
   double k=v*(0.5*v-qnorm2(delta[id]));
   double h=std::fabs(k), w=v*v;
   q.emplace_back(h);
   c[ix]=volfi::black_otm_from_variance(w,q.back());
   wt[ix]=w;
   vt[ix]=v;
   ix++;
  }
 }
 std::printf("cases %d\n",ncase);
 std::printf("repetitions_per_run %d\n",reps);
 std::printf("evaluations_per_run %d\n",reps*ncase);
 std::printf("runs %d\n",runs);
 acc("precomputed",[&](int i){return volfi::implied_variance_otm(q[i],c[i]);},wt,vt);
 std::vector<double> tp;
 for(int j=0;j<runs;j++) tp.push_back(bench([&](int i){return volfi::implied_variance_otm(q[i],c[i]);}));
 stat("precomputed",tp);
}
