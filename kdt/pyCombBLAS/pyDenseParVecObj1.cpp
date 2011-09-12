#include <mpi.h>

#include <iostream>
#include "pyDenseParVecObj1.h"

pyDenseParVecObj1::pyDenseParVecObj1()
{
}

pyDenseParVecObj1::pyDenseParVecObj1(VectType other): v(other)
{
}

pyDenseParVecObj1::pyDenseParVecObj1(int64_t size, Obj1 id): v(size, id, Obj1())
{
}

pySpParVecObj1 pyDenseParVecObj1::sparse(op::UnaryPredicateObj* keep) const
{
	if (keep != NULL)
		return pySpParVecObj1(v.Find(*keep));
	else
		return pySpParVecObj1(v.Find(bind2nd(not_equal_to<Obj1>(), Obj1())));
}

int64_t pyDenseParVecObj1::len() const
{
	return v.TotalLength();
}

int64_t pyDenseParVecObj1::__len__() const
{
	return v.TotalLength();
}
	
void pyDenseParVecObj1::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}

/*
pyDenseParVecObj1 pyDenseParVecObj1::operator==(const pyDenseParVecObj1 & rhs)
{
	//return v.operator==(rhs.v);
	pyDenseParVecObj1 ret = copy();
	ret.EWiseApply(rhs, &op::equal_to());
	return ret;
}

pyDenseParVecObj1 pyDenseParVecObj1::operator!=(const pyDenseParVecObj1 & rhs)
{
	//return !(v.operator==(rhs.v));
	pyDenseParVecObj1 ret = copy();
	ret.EWiseApply(rhs, &op::not_equal_to());
	return ret;
}*/

pyDenseParVecObj1 pyDenseParVecObj1::copy()
{
	pyDenseParVecObj1 ret;
	ret.v = v;
	return ret;
}

/////////////////////////

int64_t pyDenseParVecObj1::Count(op::UnaryPredicateObj* op)
{
	return v.Count(*op);
}

Obj1 pyDenseParVecObj1::Reduce(op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf)
{
	if (!bf->associative && root())
		cout << "Attempting to Reduce with a non-associative function! Results will be undefined" << endl;

	Obj1 ret;
	
	bf->getMPIOp();
	if (uf == NULL)
		ret = v.Reduce(*bf, Obj1(), ::identity<Obj1>());
	else
		ret = v.Reduce(*bf, Obj1(), *uf);
	bf->releaseMPIOp();
	return ret;
}

pySpParVecObj1 pyDenseParVecObj1::Find(op::UnaryPredicateObj* op)
{
	return pySpParVecObj1(v.Find(*op));
}
pySpParVecObj1 pyDenseParVecObj1::__getitem__(op::UnaryPredicateObj* op)
{
	return Find(op);
}

pyDenseParVecObj1 pyDenseParVecObj1::FindInds(op::UnaryPredicateObj* op)
{
	pyDenseParVecObj1 ret;
	
	FullyDistVec<INDEXTYPE, INDEXTYPE> fi_ret = v.FindInds(*op);
	ret.v = fi_ret;
	return ret;
}

void pyDenseParVecObj1::Apply(op::UnaryFunctionObj* op)
{
	v.Apply(*op);
}

void pyDenseParVecObj1::ApplyMasked(op::UnaryFunctionObj* op, const pySpParVec& mask)
{
	v.Apply(*op, mask.v);
}

void pyDenseParVecObj1::EWiseApply(const pyDenseParVecObj1& other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj1::EWiseApply(const pyDenseParVecObj2& other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj1::EWiseApply(const pyDenseParVec&     other, op::BinaryFunctionObj *f)
{
	v.EWiseApply(other.v, *f);
}

void pyDenseParVecObj1::EWiseApply(const pySpParVecObj1& other, op::BinaryFunctionObj *f, bool doNulls, Obj1 nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}

void pyDenseParVecObj1::EWiseApply(const pySpParVecObj2& other, op::BinaryFunctionObj *f, bool doNulls, Obj2 nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, nullValue);
}

void pyDenseParVecObj1::EWiseApply(const pySpParVec&     other, op::BinaryFunctionObj *f, bool doNulls, double nullValue)
{
	v.EWiseApply(other.v, *f, doNulls, doubleint(nullValue));
}
	
pyDenseParVecObj1 pyDenseParVecObj1::SubsRef(const pyDenseParVec& ri)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> indexv = ri.v;
	return pyDenseParVecObj1(v(indexv));
}

int64_t pyDenseParVecObj1::getnee() const
{
	return __len__();
}
/*
int64_t pyDenseParVecObj1::getnnz() const
{
	return v.Count(bind2nd(not_equal_to<doubleint>(), (double)0));
}

int64_t pyDenseParVecObj1::getnz() const
{
	return v.Count(bind2nd(equal_to<doubleint>(), (double)0));
}*/
/*
bool pyDenseParVecObj1::any() const
{
	return getnnz() > 0;
}*/

void pyDenseParVecObj1::RandPerm()
{
	v.RandPerm();
}

pyDenseParVec pyDenseParVecObj1::Sort()
{
	pyDenseParVec ret(1, 0);
	ret.v = v.sort();
	return ret; // Sort is in-place. The return value is the permutation used.
}

pyDenseParVecObj1 pyDenseParVecObj1::TopK(int64_t k)
{
	FullyDistVec<INDEXTYPE, INDEXTYPE> sel(k);
	sel.iota(k, 0);

	pyDenseParVecObj1 sorted = copy();
	//op::UnaryFunctionObj negate = op::negate();
	//sorted.Apply(&negate); // the negation is so that the sort direction is reversed
//	sorted.printall();
	FullyDistVec<INDEXTYPE, INDEXTYPE> perm = sorted.v.sort();
//	sorted.printall();
//	perm.DebugPrint();
	//sorted.Apply(&negate);

	// return dense	
	return pyDenseParVecObj1(sorted.v(sel));
}

void pyDenseParVecObj1::printall()
{
	v.DebugPrint();
}

char* pyDenseParVecObj1::__repr__()
{
	static char empty[] = {'\0'};
	printall();
	return empty;
}

Obj1 pyDenseParVecObj1::__getitem__(int64_t key)
{
	return v.GetElement(key);
}

pyDenseParVecObj1 pyDenseParVecObj1::__getitem__(const pyDenseParVec& key)
{
	return SubsRef(key);
}

void pyDenseParVecObj1::__setitem__(int64_t key, Obj1 * value)
{
	v.SetElement(key, *value);
}

/*
void pyDenseParVecObj1::__setitem__(const pySpParVec& key, const pySpParVecObj1& value)
{
	//v.Apply(::set<Obj1>(Obj1()), key.v);
	//v += value.v;
}*/

void pyDenseParVecObj1::__setitem__(const pySpParVec& key, Obj1 * value)
{
	v.Apply(::set<Obj1>(*value), key.v);
}

/*
pyDenseParVecObj1 pyDenseParVecObj1::range(int64_t howmany, int64_t start)
{
	pyDenseParVecObj1 ret;
	ret.v.iota(howmany, start-1);
	return ret;
}
*/
