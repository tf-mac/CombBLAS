//#include <mpi.h>

#include <iostream>
#include <math.h>

#include "pySpParVecObj2.h"

using namespace std;

pySpParVecObj2::pySpParVecObj2()
{
}

pySpParVecObj2::pySpParVecObj2(int64_t size): v(size)
{
}

pySpParVecObj2::pySpParVecObj2(VectType other): v(other)
{
}

pyDenseParVecObj2 pySpParVecObj2::dense() const
{
	pyDenseParVecObj2 ret(v.TotalLength(), Obj2());
	ret.v.EWiseApply(v, use2nd<Obj2>(), false, Obj2());
	return ret;
}

int64_t pySpParVecObj2::getnee() const
{
	return v.getnnz();
}

/*int64_t pySpParVecObj2::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), doubleint(0)));
}*/

int64_t pySpParVecObj2::__len__() const
{
	return v.TotalLength();
}

int64_t pySpParVecObj2::len() const
{
	return v.TotalLength();
}

pySpParVecObj2 pySpParVecObj2::copy()
{
	pySpParVecObj2 ret(0);
	ret.v = v;
	return ret;
}

bool pySpParVecObj2::any() const
{
	return getnee() != 0;
}

bool pySpParVecObj2::all() const
{
	return getnee() == v.TotalLength();
}

int64_t pySpParVecObj2::intersectSize(const pySpParVecObj2& other)
{
	cout << "intersectSize missing CombBLAS piece" << endl;
	return 0;
}

	
void pySpParVecObj2::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

void pySpParVecObj2::printall()
{
	v.DebugPrint();
}


/////////////////////////

int64_t pySpParVecObj2::Count(op::UnaryPredicateObj* op)
{
	return v.Count(*op);
}

/*
pySpParVecObj2 pySpParVecObj2::Find(op::UnaryFunctionObj* op)
{
	pySpParVecObj2 ret;
	ret->v = v.Find(*op);
	return ret;
}

pyDenseParVec pySpParVecObj2::FindInds(op::UnaryFunctionObj* op)
{
	pyDenseParVec ret = new pyDenseParVec();
	ret->v = v.FindInds(*op);
	return ret;
}
*/
void pySpParVecObj2::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}
/*
void pySpParVecObj2::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVecObj2& mask)
{
	v.Apply(*op, mask.v);
}
*/

/*
pySpParVecObj2 pySpParVecObj2::SubsRef(const pySpParVecObj2& ri)
{
	return pySpParVecObj2(v(ri.v));
}*/

pyDenseParVecObj2 pySpParVecObj2::SubsRef(const pyDenseParVec& ri)
{
	return pyDenseParVecObj2(v(ri.v));
}


Obj2 pySpParVecObj2::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	Obj2 ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, Obj2(), ::identity<Obj2>());
	else
		ret = v.Reduce(*bf, Obj2(), *uf);
	bf->releaseMPIOp();
	return ret;
}

pySpParVec pySpParVecObj2::Sort()
{
	pySpParVec ret(0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj2 pySpParVecObj2::TopK(int64_t k)
{
	//// FullyDistVec::FullyDistVec(IT glen, NT initval, NT id) 
	//FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	
	////void FullyDistVec::iota(IT globalsize, NT first)
	//sel.iota(k, v.TotalLength() - k);
	
	FullyDistSpVec<INDEXTYPE,Obj2> sorted(v);
	FullyDistSpVec<INDEXTYPE,INDEXTYPE> perm = sorted.sort();

	FullyDistVec<INDEXTYPE,INDEXTYPE> sel(k, 0, 0);
	sel.iota(k, perm.getnnz() - k);
	
	// FullyDistVec FullyDistSpVec::operator(FullyDistVec & v)
	FullyDistVec<INDEXTYPE,INDEXTYPE> topkind = perm(sel);
	cout << "perm: " << endl;
	perm.DebugPrint();
	cout << "sel: " << endl;
	sel.DebugPrint();
	cout << "topkind: " << endl;
	topkind.DebugPrint();
	
	FullyDistVec<INDEXTYPE,Obj2> topkele = v(topkind);
	//return make_pair(topkind, topkele);

	return pyDenseParVecObj2(topkele);
}

void pySpParVecObj2::setNumToInd()
{
	v.setNumToInd();
}

/*
void pySpParVecObj2::__delitem__(const pyDenseParVec& key)
{
	//v = EWiseMult(v, key.v, 1, doubleint(0));
	cout << "UNSUPORTED OP: pySpParVecObj2Obj::__delitem__(dense vector)" << endl;
}*/

void pySpParVecObj2::__delitem__(int64_t key)
{
	v.DelElement(key);
}

Obj2 pySpParVecObj2::__getitem__(int64_t key)
{
	Obj2 val = v[key];
	
	if (!v.WasFound())
	{
		//cout << "Element " << key << " not found." << endl;
	}
	
	return val;
}

Obj2 pySpParVecObj2::__getitem__(double key)
{
	return __getitem__(static_cast<int64_t>(key));
}

/*
pySpParVecObj2 pySpParVecObj2::__getitem__(const pySpParVec& key)
{
	return SubsRef(key);
}*/

pyDenseParVecObj2 pySpParVecObj2::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pySpParVecObj2::__setitem__(int64_t key, const Obj2* value)
{
	v.SetElement(key, *value);
}

void pySpParVecObj2::__setitem__(double  key, const Obj2* value)
{
	__setitem__(static_cast<int64_t>(key), value);
}

/*
void pySpParVecObj2::__setitem__(const pyDenseParVec& key, const pyDenseParVecObj2& value)
{
	if (__len__() != key.__len__())
	{
		cout << "Vector and Key different lengths" << endl;
		// throw
	}
	EWiseMult_inplacefirst(*this, key, 1, 0);
	*this += value;
}*/

void pySpParVecObj2::__setitem__(const char* key, const Obj2* value)
{
	if (strcmp(key, "existent") == 0)
	{
		v.Apply(::set<Obj2>(*value));
	}
	else
	{
		// throw
	}
}

char* pySpParVecObj2::__repr__()
{
	static char empty[] = {'\0'};
	printall();
	return empty;
}

pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVecObj2& b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls)
{
	return pySpParVecObj2(EWiseApply<Obj2>(a.v, b.v, *op, allowANulls, allowBNulls));
}

pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVecObj1& b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls)
{
	return pySpParVecObj2(EWiseApply<Obj2>(a.v, b.v, *op, allowANulls, allowBNulls));
}

pySpParVecObj2 EWiseApply(const pySpParVecObj2& a, const pySpParVec&     b, op::BinaryFunctionObj* op, bool allowANulls, bool allowBNulls)
{
	return pySpParVecObj2(EWiseApply<Obj2>(a.v, b.v, *op, allowANulls, allowBNulls));
}

/*
pySpParVecObj2 pySpParVecObj2::zeros(int64_t howmany)
{
	return pySpParVecObj2(howmany);
}*/

/*
pySpParVecObj2 pySpParVecObj2::range(int64_t howmany, int64_t start)
{
	pySpParVecObj2 ret(howmany);
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
