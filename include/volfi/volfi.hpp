#ifndef VOLFI_VOLFI_HPP
#define VOLFI_VOLFI_HPP

#include <cmath>
#include <limits>
#include <stdexcept>

#define implied_variance_atm volfi_raw_implied_variance_atm
#define implied_variance_otm volfi_raw_implied_variance_otm
#define implied_volatility_otm volfi_raw_implied_volatility_otm
#define implied_variance_call_normalised volfi_raw_implied_variance_call_normalised
#define implied_volatility_call_normalised volfi_raw_implied_volatility_call_normalised
#define implied_volatility_call volfi_raw_implied_volatility_call
#include "volfi_logc_libm.hpp"
#undef implied_variance_atm
#undef implied_variance_otm
#undef implied_volatility_otm
#undef implied_variance_call_normalised
#undef implied_volatility_call_normalised
#undef implied_volatility_call

namespace volfi {
#include "volfi_fastpatch.hpp"

inline double domain_error_value(){
#ifdef VOLFI_STRICT_DOMAIN
 throw std::domain_error("volfi domain error");
#else
 return std::numeric_limits<double>::quiet_NaN();
#endif
}

inline bool open_unit(double x){return std::isfinite(x) && x>0.0 && x<1.0;}
inline bool pos_finite(double x){return std::isfinite(x) && x>0.0;}
inline double normalised_call_lower(double k){return k<0.0 ? 1.0-std::exp(k) : 0.0;}
inline bool valid_normalised_call(double k,double c){return std::isfinite(k) && open_unit(c) && c>normalised_call_lower(k);}
inline bool valid_otm(double h,double c){return std::isfinite(h) && h>0.0 && open_unit(c);}

inline double implied_variance_atm(double c){
 if(!open_unit(c)) return domain_error_value();
 return volfi_raw_implied_variance_atm(c);
}

inline double implied_variance_otm_exact(const otm_context& q,double c){
 if(!valid_otm(q.h,c)) return domain_error_value();
 return volfi_raw_implied_variance_otm(q,c);
}

inline double implied_variance_otm_fast(const otm_context& q,double c){
 if(!valid_otm(q.h,c)) return domain_error_value();
 double w=seed_otm(q,c), wf;
 if(fastpatch_variance(w,q,c,wf)) return wf;
 return halley_variance(w,q,c);
}

inline double implied_variance_otm(const otm_context& q,double c){
#ifdef VOLFI_ENABLE_FASTPATCH
 return implied_variance_otm_fast(q,c);
#else
 return implied_variance_otm_exact(q,c);
#endif
}

inline double implied_variance_otm_exact(double h,double c){
 if(!valid_otm(h,c)) return domain_error_value();
 otm_context q(h);
 return implied_variance_otm_exact(q,c);
}

inline double implied_variance_otm_fast(double h,double c){
 if(!valid_otm(h,c)) return domain_error_value();
 otm_context q(h);
 return implied_variance_otm_fast(q,c);
}

inline double implied_variance_otm(double h,double c){
 if(!valid_otm(h,c)) return domain_error_value();
 otm_context q(h);
 return implied_variance_otm(q,c);
}

inline double implied_volatility_otm(const otm_context& q,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_otm(q,c)/t);
}

inline double implied_volatility_otm_fast(const otm_context& q,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_otm_fast(q,c)/t);
}

inline double implied_volatility_otm(double h,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_otm(h,c)/t);
}

inline double implied_volatility_otm_fast(double h,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_otm_fast(h,c)/t);
}

inline double implied_variance_call_normalised(double k,double c){
 if(!valid_normalised_call(k,c)) return domain_error_value();
 if(k>0.0) return implied_variance_otm(k,c);
 if(k<0.0){
  double cp=1.0+std::exp(-k)*(c-1.0);
  if(!open_unit(cp)) return domain_error_value();
  return implied_variance_otm(-k,cp);
 }
 return implied_variance_atm(c);
}

inline double implied_variance_call_normalised_fast(double k,double c){
 if(!valid_normalised_call(k,c)) return domain_error_value();
 if(k>0.0) return implied_variance_otm_fast(k,c);
 if(k<0.0){
  double cp=1.0+std::exp(-k)*(c-1.0);
  if(!open_unit(cp)) return domain_error_value();
  return implied_variance_otm_fast(-k,cp);
 }
 return implied_variance_atm(c);
}

inline double implied_volatility_call_normalised(double k,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_call_normalised(k,c)/t);
}

inline double implied_volatility_call_normalised_fast(double k,double c,double t){
 if(!pos_finite(t)) return domain_error_value();
 return std::sqrt(implied_variance_call_normalised_fast(k,c)/t);
}

inline double implied_volatility_call(double f,double k,double d,double t,double price){
 if(!pos_finite(f) || !pos_finite(k) || !pos_finite(d) || !pos_finite(t) || !std::isfinite(price)) return domain_error_value();
 double kk=std::log(k/f), c=price/(d*f);
 return implied_volatility_call_normalised(kk,c,t);
}

inline double implied_volatility_call_fast(double f,double k,double d,double t,double price){
 if(!pos_finite(f) || !pos_finite(k) || !pos_finite(d) || !pos_finite(t) || !std::isfinite(price)) return domain_error_value();
 double kk=std::log(k/f), c=price/(d*f);
 return implied_volatility_call_normalised_fast(kk,c,t);
}

}

#endif
