#include <iostream>
#include <fstream>
#include <mpi.h>
#include <sys/time.h>
#include "SpTuples.h"
#include "SpDCCols.h"
#include "SpParMPI2.h"

using namespace std;

int main()
{
	MPI::Init();
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	typedef PlusTimesSRing<double, double> PT;	

	if (myrank == 0)
	{
		ifstream inputa("matrixA.txt");
		ifstream inputb("matrixB.txt");
		if(!(inputa.is_open() && inputb.is_open()))
		{
			cerr << "One of the input files doesn't exist\n";
			exit(-1);
		}
	
		SpDCCols<int,double> A;
		inputa >> A;
		A.PrintInfo();
	
		SpDCCols<int,double> B;
		inputb >> B;		
		B.PrintInfo();

		// arguments (in this order): nnz, n, m, nzc
		SpDCCols<int,double> C(0, A.getnrow(), B.getncol(), 0);
		C.PrintInfo();

		C.SpGEMM <PT> (A, B, false, false);	// C = A*B
		C.PrintInfo();

		SpDCCols<int,bool> A_bool = A.ConvertNumericType<bool>();
		A_bool.PrintInfo();

		SpTuples<int,double> C_tuples =  MultiplyReturnTuples<PT>(A_bool, B, false, false);	// D = A_bool*B
		C_tuples.PrintInfo();

		SpTuples<int,double> C_tt = MultiplyReturnTuples<PT>(B, A_bool, false, true);
		C_tt.PrintInfo();
	}

	#define BIGTEST
	//#define MASSIVETEST
		// Start big timing test
		vector<string> prefixes;

	#ifdef BIGTEST
		prefixes.push_back("largeseq");
	#endif
	#ifdef MASSIVETEST
		prefixes.push_back("massiveseq");
	#endif
	
	for(int i=0; i< prefixes.size(); i++)
	{
		ifstream input1, input2;
		if(myrank == 0)
		{ 
			string inputname1 = prefixes[i] + string("/input1_0");
			string inputname2 = prefixes[i] + string("/input2_0");
			input1.open(inputname1.c_str());
        		input2.open(inputname2.c_str());
        		if(!(input1.is_open() && input2.is_open()))
        		{
                		cerr << "One of the input files doesn't exist\n";
				exit(-1);
        		}
			SpDCCols<int,double> bigA;
        		input1 >> bigA;
        		bigA.PrintInfo();

			SpDCCols<int,double> bigA1, bigA2;
			bigA.Split(bigA1, bigA2);
			bigA1.PrintInfo();
			bigA2.PrintInfo();

			bigA.Merge(bigA1, bigA2);
			bigA.PrintInfo();

       		 	SpDCCols<int,double> bigB;
        		input2 >> bigB;
        		bigB.PrintInfo();

			// Cache warm-up
			SpTuples<int,double> bigC = MultiplyReturnTuples<PT>(bigA, bigB, false, false);
			bigC.PrintInfo();

	#ifdef OUTPUT
			string outputnameC = prefixes[i] + string("/colbycol");
			ofstream outputC(outputnameC.c_str());
			outputC << bigC;
			outputC.close();
	#endif
			struct timeval tempo1, tempo2;
	
			double elapsed_time;    /* elapsed time in seconds */
			long elapsed_seconds;  /* diff between seconds counter */
			long elapsed_useconds; /* diff between microseconds counter */

			gettimeofday(&tempo1, NULL);
			bigC = MultiplyReturnTuples<PT>(bigA, bigB, false, false);
			gettimeofday(&tempo2, NULL);
			elapsed_seconds  = tempo2.tv_sec  - tempo1.tv_sec;
			elapsed_useconds = tempo2.tv_usec - tempo1.tv_usec;

			elapsed_time = (elapsed_seconds + ((double) elapsed_useconds)/1000000.0);
			printf("ColByCol time = %.5f seconds\n", elapsed_time);

			bigB.Transpose();	// now that bigB is transposed, bigC_t will be equal to bigC
			bigB.PrintInfo();
	
			// Cache warm-up
			SpTuples<int,double> bigC_t = MultiplyReturnTuples<PT>(bigA, bigB, false, true);
			bigC_t.PrintInfo();

	#ifdef OUTPUT	
			string outputnameCT = prefixes[i] + string("/outerproduct");
			ofstream outputC(outputnameCT.c_str());
			outputCT << bigC_t;
			outputCT.close();
	#endif

			gettimeofday(&tempo1, NULL);
			bigC_t = MultiplyReturnTuples<PT>(bigA, bigB, false, true);
			gettimeofday(&tempo2, NULL);
			elapsed_seconds  = tempo2.tv_sec  - tempo1.tv_sec;
			elapsed_useconds = tempo2.tv_usec - tempo1.tv_usec;
		
			elapsed_time = (elapsed_seconds + ((double) elapsed_useconds)/1000000.0);
			printf("OuterProduct time = %.5f seconds\n", elapsed_time);
	
			input1.seekg (0, ios::beg);
			input2.seekg (0, ios::beg);
	
			cerr << "Begin Parallel" << endl;	
		}	
		// ABAB: Make a macro such as "PARTYPE(it,nt,seqtype)" that just typedefs this guy !
		typedef SpParMPI2 <int, double, SpDCCols<int,double> > PARSPMAT;
		PARSPMAT A_par;
		PARSPMAT B_par;
		cerr << "A and B constructed"<< endl;

		A_par.ReadDistribute(input1, 0);
		B_par.ReadDistribute(input2, 0);
		
		/* collective calls
		int parnnzA = A_par.getnnz();
		int parmA = A_par.getnrow();
		int parnA = A_par.getncol();

		int parnnzB = B_par.getnnz();
		int parmB = B_par.getnrow();
		int parnB = B_par.getncol();
		
		if(myrank == 0)
		{
			cout << "A_par has " << parnnzA << " nonzeros and " << parmA << "-by-" << parnA << " dimensions" << endl;
			cout << "B_par has " << parnnzB << " nonzeros and " << parmB << "-by-" << parnB << " dimensions" << endl;
		}
		*/

		//PARSPMAT C_par = Mult_AnXBn<PT> (A_par, B_par);	

		if(myrank == 0)
		{
			input1.close();	
			input2.close();
		}
	}
	MPI::Finalize();
}
