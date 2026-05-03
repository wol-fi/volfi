#include <Rcpp.h>
#include "volfi.hpp"
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector iv(NumericVector k, NumericVector c){
 int n=std::max(k.size(),c.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::implied_variance_call_normalised(k[i%k.size()],c[i%c.size()]);
 return out;
}

// [[Rcpp::export]]
NumericVector vol(NumericVector k, NumericVector c, NumericVector t){
 int n=std::max(std::max(k.size(),c.size()),t.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=std::sqrt(volfi::implied_variance_call_normalised(k[i%k.size()],c[i%c.size()])/t[i%t.size()]);
 return out;
}

// [[Rcpp::export]]
NumericVector iv_atm(NumericVector c){
 int n=c.size();
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::implied_variance_atm(c[i]);
 return out;
}

// [[Rcpp::export]]
NumericVector black(NumericVector w, NumericVector h){
 int n=std::max(w.size(),h.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::black_otm_from_variance(w[i%w.size()],std::abs(h[i%h.size()]));
 return out;
}
