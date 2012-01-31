#ifndef PY_SP_PAR_MAT_H
#define PY_SP_PAR_MAT_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParMat {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

public:
	typedef int64_t INDEXTYPE;
	typedef doubleint NUMTYPE;
	typedef SpDCCols<INDEXTYPE,NUMTYPE> DCColsType;
	typedef SpDCCols<INDEXTYPE,double> DCColsTypeDouble;
	typedef SpParMat < INDEXTYPE, NUMTYPE, DCColsType > PSpMat_DoubleInt;
	typedef SpParMat < INDEXTYPE, double, DCColsType > PSpMat_Double;
	typedef PSpMat_DoubleInt MatType;
	
public:
	
	pySpParMat(MatType other);
	pySpParMat(const pySpParMat& copyFrom);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMat();
	pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);

	pySpParMat(const pySpParMatBool& copyStructureFrom);
	pySpParMat(const pySpParMatObj1& copyStructureFrom);
	pySpParMat(const pySpParMatObj2& copyStructureFrom);
	
public:
	int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR = 16, bool delIsolated=true, double a=.57, double b=.19, double c=.19, double d=.05);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat copy();
	pySpParMat& operator+=(const pySpParMat& other);
	pySpParMat& assign(const pySpParMat& other);
	pySpParMat SpGEMM(pySpParMat& other, op::Semiring* sring = NULL);
	//pySpParMat SpGEMM(pySpParMat& other, op::SemiringObj* sring = NULL);
	pySpParMat operator*(pySpParMat& other);
	pySpParMat SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols, bool inPlace = false);
	pySpParMat __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunction* f);
	void Apply(op::UnaryFunctionObj* f);
	void DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunction* f);
	void DimWiseApply(int dim, const pyDenseParVec& values, op::BinaryFunctionObj* f);
	pySpParMat Prune(op::UnaryFunction* f, bool inPlace);
	pySpParMat Prune(op::UnaryPredicateObj* f, bool inPlace);
	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVec Reduce(int dim, op::BinaryFunction* f, double identity = 0);
	pyDenseParVec Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity = 0);
	void Reduce(int dim, pyDenseParVec* ret, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, double identity = 0);

	void Transpose();
	//void EWiseMult(pySpParMat* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:

	pySpParVec SpMV(const pySpParVec& x, op::Semiring* sring);
	pyDenseParVec SpMV(const pyDenseParVec& x, op::Semiring* sring);
	void SpMV_inplace(pySpParVec& x, op::Semiring* sring);
	void SpMV_inplace(pyDenseParVec& x, op::Semiring* sring);

	pySpParVec SpMV(const pySpParVec& x, op::SemiringObj* sring);
	pyDenseParVec SpMV(const pyDenseParVec& x, op::SemiringObj* sring);
	void SpMV_inplace(pySpParVec& x, op::SemiringObj* sring);
	void SpMV_inplace(pyDenseParVec& x, op::SemiringObj* sring);

	void Square(op::Semiring* sring);
	void Square(op::SemiringObj* sring);
	pySpParMat     SpGEMM(pySpParMat     &other, op::SemiringObj* sring);
	//pySpParMatBool SpGEMM(pySpParMatBool &other, op::SemiringObj* sring);
	/* CombBLAS support for these is still not there.
	pySpParMatObj1 SpGEMM(pySpParMatObj1 &other, op::SemiringObj* sring);
	pySpParMatObj2 SpGEMM(pySpParMatObj2 &other, op::SemiringObj* sring);
	*/
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);
pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat& B, op::BinaryFunction *bf, bool notB = false, double defaultBValue = 1);

pySpParMat EWiseApply(const pySpParMat& A, const pySpParMatObj1& B, op::BinaryFunctionObj *bf, bool notB = false, Obj1 defaultBValue = Obj1());
pySpParMat EWiseApply(const pySpParMat& A, const pySpParMatObj2& B, op::BinaryFunctionObj *bf, bool notB = false, Obj2 defaultBValue = Obj2());
pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat&     B, op::BinaryFunctionObj *bf, bool notB = false, double defaultBValue = 0);

pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat&     B, op::BinaryFunctionObj* op, op::BinaryPredicateObj* doOp, bool allowANulls, bool allowBNulls, double ANull, double BNull);

//INTERFACE_INCLUDE_END


// From CombBLAS/promote.h:
/*
template <class T1, class T2>
struct promote_trait  { };

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
*/
DECLARE_PROMOTE(pySpParMat::MatType, pySpParMat::MatType, pySpParMat::MatType)
DECLARE_PROMOTE(pySpParMat::DCColsType, pySpParMat::DCColsType, pySpParMat::DCColsType)

template <> struct promote_trait< SpDCCols<int64_t,doubleint> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;                    
    };

template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,doubleint> >       
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;                    
    };

// Based on what's in CombBLAS/SpDCCols.h:
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, doubleint> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };


#endif
