/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_TUPLES_H
#define _SP_TUPLES_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>

#ifdef NOTR1
	#include <boost/tr1/tuple.hpp>
#else
	#include <tr1/tuple>
#endif
#include "SpMat.h"
#include "SpDefs.h"
#include "StackEntry.h"
#include "Compare.h"

using namespace std;
using namespace std::tr1;

template <class IU, class NU>
class SpDCCols;

template <class IU, class NU>
class Dcsc;

#ifdef GRAPHGEN_KEEP_MULTIPLICITIES
typedef struct generated_edge {
  int64_t src;
  int64_t tgt;
  int64_t multiplicity;
} generated_edge;
#endif

/**
 * Triplets are represented using the boost::tuple class of the Boost library
 * \remarks Indices start from 0 in this class
 */
template <class IT, class NT>
class SpTuples: public SpMat<IT, NT, SpTuples<IT,NT> >
{
public:
	// Constructors 
	SpTuples (IT size, IT nRow, IT nCol);
	SpTuples (IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);
	SpTuples (IT maxnnz, IT nRow, IT nCol, IT * edges);	// Graph500 contructor
	SpTuples (IT size, IT nRow, IT nCol, StackEntry<NT, pair<IT,IT> > * & multstack);		
	SpTuples (const SpTuples<IT,NT> & rhs);	 	// Actual Copy constructor
	SpTuples (const SpDCCols<IT,NT> & rhs); 	// Copy constructor for conversion from SpDCCols
	~SpTuples();

	SpTuples<IT,NT> & operator=(const SpTuples<IT,NT> & rhs);

	IT & rowindex (IT i) { return tr1::get<0>(tuples[i]); }
	IT & colindex (IT i) { return tr1::get<1>(tuples[i]); }
	NT & numvalue (IT i) { return tr1::get<2>(tuples[i]); }

	IT rowindex (IT i) const { return tr1::get<0>(tuples[i]); }
	IT colindex (IT i) const { return tr1::get<1>(tuples[i]); } 
	NT numvalue (IT i) const { return tr1::get<2>(tuples[i]); } 

	void SortRowBased()
	{
		if(!SpHelper::is_sorted(tuples, tuples+nnz))
			sort(tuples , tuples+nnz);	// Default "operator<" for tuples uses lexicographical ordering 
	}

	void SortColBased()
	{
		ColLexiCompare<IT,NT> collexicogcmp;
		if(!SpHelper::is_sorted(tuples, tuples+nnz, collexicogcmp))
			sort(tuples , tuples+nnz, collexicogcmp );
	}

	pair<IT,IT> RowLimits()
	{
		if(nnz > 0)
		{	
			RowCompare<IT,NT> rowcmp;
			tuple<IT,IT,NT> * maxit = max_element(tuples, tuples+nnz, rowcmp);	
			tuple<IT,IT,NT> * minit = min_element(tuples, tuples+nnz, rowcmp);
			return make_pair(tr1::get<0>(*minit), tr1::get<0>(*maxit));
		}
		else
			return make_pair(zero,zero);
	}
	pair<IT,IT> ColLimits()
	{	
		if(nnz > 0)
		{
			ColCompare<IT,NT> colcmp;
			tuple<IT,IT,NT> * maxit = max_element(tuples, tuples+nnz, colcmp);
			tuple<IT,IT,NT> * minit = min_element(tuples, tuples+nnz, colcmp);
			return make_pair(tr1::get<1>(*minit), tr1::get<1>(*maxit));
		}
		else
			return make_pair(zero,zero);
	}

	// Performs a balanced merge of the array of SpTuples
	template<typename SR, typename IU, typename NU>
	friend SpTuples<IU,NU> MergeAll(const vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar, IU nstar); 

	template<typename SR, typename IU, typename NU>
	friend SpTuples<IU,NU> * MergeAllRec(const vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar, IU nstar); 
	
	ofstream& put (ofstream& outfile) const;		
	ifstream& get (ifstream& infile); 

	bool isZero() const { return (nnz == zero); }	
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }

	void PrintInfo();

private:
	tuple<IT, IT, NT> * tuples; 	// boost:tuple
	/** 
	 **	tuple elements with indices:
	 **	0) IT * ir ;	    	//  array of row indices, size nnz 
	 **	1) IT * jc ;	    	//  array of col indices, size nnz 
	 **	2) NT * numx;		//  array of generic values, size nnz
	 **/

	IT m;
	IT n;
	IT nnz;	

	const static IT zero;	

	SpTuples (){};		// Default constructor does nothing, hide it
	
	void FillTuples (Dcsc<IT,NT> * mydcsc);

	template <class IU, class NU>
	friend class SpDCCols;
};


// At this point, complete type of of SpTuples is known, safe to declare these specialization (but macros won't work as they are preprocessed)
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,unsigned> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,unsigned> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };


#include "SpTuples.cpp"	
#endif
