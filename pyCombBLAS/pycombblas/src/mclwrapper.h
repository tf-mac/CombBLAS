//
// Created by exouser on 4/26/25.
//

#ifndef MCLWRAPPER_H
#define MCLWRAPPER_H

namespace combblas
{
template <typename IT, typename NT, typename DER>
std::vector<IT> pyCC(SpParMat<IT,NT,DER> & A){
    IT nCC;
    FullyDistVec<IT, IT> cclabels = CC(A, nCC);
    std::vector<IT> globalcclabels;
    SpParHelper::AllgatherVector(globalcclabels, cclabels.GetLocVec());
    return globalcclabels;
}

template <typename IT, typename NT, typename DER>
void MakeColStochastic(SpParMat<IT,NT,DER> & A)
{
    FullyDistVec<IT, NT> colsums = A.Reduce(Column, std::plus<NT>(), 0.0);
    colsums.Apply(safemultinv<NT>());
    A.DimApply(Column, colsums, std::multiplies<NT>());    // scale each "Column" with the given vector
}

template <typename IT, typename NT, typename DER>
NT Chaos(SpParMat<IT,NT,DER> & A)
{
    // sums of squares of columns
    FullyDistVec<IT, NT> colssqs = A.Reduce(Column, std::plus<NT>(), 0.0, std::bind(exponentiate(),std::placeholders::_1, 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), 0.0);
    colmaxs -= colssqs;
    // multiplu by number of nonzeros in each column
    FullyDistVec<IT, NT> nnzPerColumn = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, std::multiplies<NT>());
    return colmaxs.Reduce(maximum<NT>(), 0.0);
}

template <typename IT, typename NT, typename DER>
void Inflate(SpParMat<IT,NT,DER> & A, double power)
{
    A.Apply(std::bind(exponentiate(),std::placeholders::_1, power));
}


// default adjustloop setting
// 1. Remove loops
// 2. set loops to max of all arc weights
template <typename IT, typename NT, typename DER>
void AdjustLoops(SpParMat<IT,NT,DER> & A)
{

    A.RemoveLoops();
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), std::numeric_limits<NT>::min());
    A.Apply([](NT val){return val==std::numeric_limits<NT>::min() ? 1.0 : val;}); // for isolated vertices
    A.AddLoops(colmaxs);
    std::ostringstream outs;
    outs << "Adjusting loops" << endl;
    //SpParHelper::Print(outs.str());
}

template <typename IT, typename NT, typename DER>
void RemoveIsolated(SpParMat<IT,NT,DER> & A)
{
    std::ostringstream outs;
    FullyDistVec<IT, NT> ColSums = A.Reduce(Column, std::plus<NT>(), 0.0);
    FullyDistVec<IT, IT> nonisov = ColSums.FindInds(std::bind(std::greater<NT>(),std::placeholders::_1, 0));
    IT numIsolated = A.getnrow() - nonisov.TotalLength();
    outs << "Number of isolated vertices: " << numIsolated << endl;
    //SpParHelper::Print(outs.str());
    A(nonisov, nonisov, true);
    //SpParHelper::Print("Removed isolated vertices.\n");
    A.PrintInfo();
}

template <typename IT, typename NT, typename DER>
std::vector<NT> ColumnNNZ(SpParMat<IT,NT,DER> & A)
{
    std::ostringstream outs;
    FullyDistVec<IT, NT> ColSums = A.Reduce(Column, std::plus<NT>(), 0.0, [](NT val){return 1.0;});
    std::vector<NT> ret;
    SpParHelper::AllgatherVector(ret, ColSums.GetLocVec());
    return ret;
}



//TODO: handle reordered cluster ids
template <typename IT, typename NT, typename DER>
void RandPermute(SpParMat<IT,NT,DER> & A)
{
    // randomly permute for load balance
    if(A.getnrow() == A.getncol())
    {
        FullyDistVec<IT, IT> p( A.getcommgrid());
        p.iota(A.getnrow(), 0);
        p.RandPerm();
        (A)(p,p,true);// in-place permute to save memory
        // //SpParHelper::Print("Applied symmetric permutation.\n");
    }
    else
    {
        //SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
    }
}

//TODO: handle reordered cluster ids
template <typename IT, typename NT, typename DER>
void PermuteSpParMat(SpParMat<IT,NT,DER> & A, const std::vector<IT> & permvector)
{
    // randomly permute for load balance
    if(A.getnrow() == A.getncol())
    {
        // permvector is either in the first process or distributed across the process.
        FullyDistVec<IT, IT> p(permvector, A.getcommgrid());
        (A)(p,p,true);// in-place permute to save memory
        // //SpParHelper::Print("Applied symmetric permutation.\n");
    }
    else
    {
        //SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
    }
}

template <typename IT, typename NT, typename DER>
void Symmetricize(SpParMat<IT,NT,DER> & A)
{
    SpParMat<IT,NT,DER> AT = A;
    AT.Transpose();
    if(!(AT == A))
    {
        //SpParHelper::Print("Symmatricizing an unsymmetric input matrix.\n");
        A += AT;
    }else{
        //SpParHelper::Print("input matrix is symmetric.\n");
    }
}
}


#endif //MCLWRAPPER_H
