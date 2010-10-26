#ifndef _DENSE_PAR_VEC_H_
#define _DENSE_PAR_VEC_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <iterator>

#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif
#include "CommGrid.h"

using namespace std;
using namespace std::tr1;

template <class IT, class NT>
class SpParVec;

template <class IT, class NT, class DER>
class SpParMat;

template <class IT, class NT>
class DenseParVec
{
public:
	DenseParVec ( );
	DenseParVec ( shared_ptr<CommGrid> grid, NT id);
	bool operator== (const DenseParVec<IT,NT> & rhs) const;
	ifstream& ReadDistribute (ifstream& infile, int master);
	DenseParVec<IT,NT> &  operator=(const SpParVec<IT,NT> & rhs);		//!< SpParVec->DenseParVec conversion operator
	DenseParVec<IT,NT> & operator+=(const DenseParVec<IT,NT> & rhs);
	DenseParVec<IT,NT> & operator-=(const DenseParVec<IT,NT> & rhs);

	bool operator==(const DenseParVec<IT,NT> & rhs)
	{
		ErrorTolerantEqual<NT> epsilonequal;
		int local = 1;
		if(diagonal)
		{
			local = (int) std::equal(arr.begin(), arr.end(), rhs.arr.begin(), epsilonequal );
		#ifdef DEBUG
			vector<NT> diff(arr.size());
			transform(arr.begin(), arr.end(), rhs.arr.begin(), diff.begin(), minus<NT>());
			typename vector<NT>::iterator maxitr;
			maxitr = max_element(diff.begin(), diff.end()); 			
			cout << maxitr-diff.begin() << ": " << *maxitr << " where lhs: " << *(arr.begin()+(maxitr-diff.begin())) 
							<< " and rhs: " << *(rhs.arr.begin()+(maxitr-diff.begin())) << endl; 
			if(local == 0)
			{
				PrintToFile("y");
			}
		#endif
		}
		int whole = 1;
		commGrid->GetWorld().Allreduce( &local, &whole, 1, MPI::INT, MPI::BAND);
		return static_cast<bool>(whole);	
	}

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{	
		transform(arr.begin(), arr.end(), arr.begin(), __unary_op);
	}	
	
	void PrintToFile(string prefix)
	{
		ofstream output;
		commGrid->OpenDebugFile(prefix, output);
		copy(arr.begin(), arr.end(), ostream_iterator<NT> (output, " "));
		output << endl;
		output.close();
	}
	
	template <typename _BinaryOperation>
	NT Reduce(_BinaryOperation __binary_op, NT identity);	// ABAB: What's the purpose of this function?
			
private:
	shared_ptr<CommGrid> commGrid;
	vector< NT > arr;
	bool diagonal;
	NT zero;	//!< the element for non-existings scalars (0.0 for a vector on Reals, +infinity for a vector on the tropical semiring) 

	template <typename _BinaryOperation>	
	void EWise(const DenseParVec<IT,NT> & rhs,  _BinaryOperation __binary_op);

	template <class IU, class NU>
	friend class DenseParMat;

	template <class IU, class NU, class UDER>
	friend class SpParMat;

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend DenseParVec<IU,typename promote_trait<NUM,NUV>::T_promote> 
	SpMV (const SpParMat<IU,NUM,UDER> & A, const DenseParVec<IU,NUV> & x );
};

#include "DenseParVec.cpp"
#endif


