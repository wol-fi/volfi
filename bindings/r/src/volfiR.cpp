#include <Rcpp.h>
#include <volfi/volfi.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector iv_otm(NumericVector h, NumericVector c){
 int n=std::max(h.size(),c.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::implied_variance_otm(h[i%h.size()],c[i%c.size()]);
 return out;
}

// [[Rcpp::export]]
NumericVector vol_otm(NumericVector h, NumericVector c, NumericVector t){
 int n=std::max(std::max(h.size(),c.size()),t.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::implied_volatility_otm(volfi::otm_context(h[i%h.size()]),c[i%c.size()],t[i%t.size()]);
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
NumericVector black_otm(NumericVector w, NumericVector h){
 int n=std::max(w.size(),h.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::black_otm_from_variance(w[i%w.size()],h[i%h.size()]);
 return out;
}

// [[Rcpp::export]]
NumericVector iv_call_norm(NumericVector k, NumericVector c){
 int n=std::max(k.size(),c.size());
 NumericVector out(n);
 for(int i=0;i<n;i++) out[i]=volfi::implied_variance_call_normalised(k[i%k.size()],c[i%c.size()]);
 return out;
}
