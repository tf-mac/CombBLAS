#ifndef NSPARSE_H
#define NSPARSE_H

#ifdef USE_CUDA

#include <time.h>

#include <random>
#include <typeinfo>

#include "utils/CSR.h"
#include "utils/def.h"

namespace nsparse
{

template <class idType, class valType>
void get_spgemm_flop(CSR<idType, valType> a, CSR<idType, valType> b, long long int& flop);

template <class idType, class valType>
void SpGEMM_Hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType>& c);

template <class idType, class valType>
void SpGEMM_Hash_Numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType>& c);

}  // namespace nsparse

#endif

#endif
