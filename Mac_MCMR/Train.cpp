/* This file contains code for mcmrtrain. It takes as command line arguments, the train file, and outputs the model file. */

#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include "string.h"
#include "soplex-1.7.2/src/soplex.h"

using namespace soplex;

#define EPS 1/infinity 
#define INF infinity 


/* This is an auxiliary function for timekeeping */
double diffclock(int clock1,int clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(float(diffticks))/CLOCKS_PER_SEC;
	return diffms;
}
/***********************************/


/* This defines the kernel used */
inline float kernel(const int flag,const float Gamma, const float *x1, const float *x2, const int D)
{
	if(flag == 0) 
	{
		/* Linear Kernel */
		float sum=0.0;
		for (int d=0;d<D;d++)
		sum += x1[d]*x2[d];
		return sum;
	}
	else if(flag == 1)
	{
		/* RBF Kernel */
		float sqrNorm=0.0;
		for (int d=0; d<D; d++)
		{
		sqrNorm += pow(x1[d]-x2[d],2);
		}
		return exp(-Gamma*sqrNorm);
	}
	else
	{
		std::cout<<"Improper choice for kernel";
		return 0;
	}		
}
/***********************************/


/* This function performs the training by solving the requisite linear program of MCM */

void trainMCM(float *wPtr, const float *xTrain, const float *yTrain, const int N, const int D, const float C, const int kernel_flag, const float Gamma, float epsilon)
{

/* Construct the problem here */

	SoPlex prob;

	/* Length of variable vector. Number of constraints */
	//	lambda	b	Q+	Q-		H		P
	int numVars = N + 1 + 2*N + 1 + 1, numConstr = 4*N;

	/* set the objective sense */
	prob.changeSense(SPxLP::MINIMIZE);

	/* we first add variables */
	DSVector dummycol(0);


//	int begintr=clock();  

	/* Now we construct the objective of the LP using SoPlex syntax */
	int v=0;

	// lambda
	while(v<N)
	{
		prob.addCol(LPCol(0.0, dummycol, INF, -INF)); 
		v++;
	}

	//b
	prob.addCol(LPCol(0.0, dummycol, INF, -INF));
	v++;

	//Q+,Q-
	while(v<3*N+1)
	{
		prob.addCol(LPCol(C, dummycol, INF, 0.0)); 
		v++;
	}

	//H
	prob.addCol(LPCol(1.0, dummycol, INF, 0.0));
	v++;

	//P
	prob.addCol(LPCol(0.0, dummycol, INF, -INF));



	/* Add constraints one by one */

	/* First Constraint */
	// (lambda)*K(xi,xj) +b +p(yi + epsilon) - H <= 0

	int c=0;
	while (c<N)
	{
		DSVector row(numVars);
		int v=0;

		//lambda
		while (v<N)
		{
			float temp = kernel(kernel_flag, Gamma, xTrain+v*D, xTrain+c*D, D);
			row.add(v, temp);
			v++;
		}

		//b
		row.add(v, 1.0);
		v++;

		//Q+
		while(v<2*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//Q-
		while(v<3*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//H
		row.add(3*N+1, -1.0);

		//P
		row.add(3*N+2, yTrain[c]+epsilon);


		prob.addRow(LPRow(-INF, row, 0.0));
		c++;
	}



	/* Second constraint */
	// -(lambda)*K(xi,xj) -b -p(yi - epsilon) -H <= 0
	while (c<2*N)
	{
		DSVector row(numVars);
		int v=0;

		//lambda
		while (v<N)
		{
			float temp = kernel(kernel_flag, Gamma, xTrain+v*D, xTrain+(c-N)*D, D);
			row.add(v, -temp);
			v++;
		}

		//b
		row.add(v, -1.0);
		v++;

		//Q+
		while(v<2*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//Q-
		while(v<3*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//H
		row.add(3*N+1, -1.0);

		//P
		row.add(3*N+2, -yTrain[(c-N)]+epsilon);


		prob.addRow(LPRow(-INF, row, 0.0));
		c++;
	}



	/* Third constraint */
	// -(lambda)*K(xi,xj) - b - p(yi + epsilon) - qi+  <= -1
	while (c<3*N)
	{
		DSVector row(numVars);
		int v=0;

		//lambda
		while (v<N)
		{
			float temp = kernel(kernel_flag, Gamma, xTrain+v*D, xTrain+(c-2*N)*D, D);
			row.add(v, -temp);
			v++;
		}

		//b
		row.add(v, -1.0);
		v++;

		//Q+
		while(v<2*N+1)
		{
			if(2*N+1-v == 3*N - c)
				row.add(v, -1.0);
			else
				row.add(v, 0.0);
			v++;
		}

		//Q-
		while(v<3*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//H
		row.add(3*N+1, 0.0);

		//P
		row.add(3*N+2, -yTrain[(c-2*N)]-epsilon);


		prob.addRow(LPRow(-INF, row, -1.0));
		c++;
	}


	/* Fourth constraint */
	// (lambda)*K(xi,xj) + b + p(yi - epsilon) - qi-  <= -1
	while (c<4*N)
	{
		DSVector row(numVars);
		int v=0;

		//lambda
		while (v<N)
		{
			float temp = kernel(kernel_flag, Gamma, xTrain+v*D, xTrain+(c-3*N)*D, D);
			row.add(v, +temp);
			v++;
		}

		//b
		row.add(v, 1.0);
		v++;

		//Q+
		while(v<2*N+1)
		{
			row.add(v, 0.0);
			v++;
		}

		//Q-
		while(v<3*N+1)
		{
			if(3*N+1-v == 4*N - c)
				row.add(v, -1.0);
			else
				row.add(v, 0.0);
			v++;
		}

		//H
		row.add(3*N+1, 0.0);

		//P
		row.add(3*N+2, +yTrain[(c-3*N)]-epsilon);


		prob.addRow(LPRow(-INF, row, -1.0));
		c++;
	}


//	int endtr=clock();

	/* solve LP */
	SPxSolver::Status stat;
	DVector prim(numVars);
	DVector dual(numConstr);
	 stat = prob.solve();

	/* get solution */
	if( stat == SPxSolver::OPTIMAL )
	{
		prob.getPrimal(prim);
		prob.getDual(dual);
		std::cout << "LP solved to optimality.\n";
		std::cout << "Objective value is " << prob.objValue() << ".\n";
		//store objective value in results file
		float Opt = prob.objValue();


		for (int v=0; v<numVars; v++)
		{
		wPtr[v] = (float)prim[v];
		}
	}

}

/***********************************/


/* Auxiliary string functions */

/* Split string by delimiter */
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

/***********************************/

/* Convert string to float */
float strtofloat(std::string s)
{
	return atof(s.c_str());
}


/* Convert string to int */
int strtoint(std::string s)
{
	return atoi(s.c_str());
}

/* Convert int to string */
std::string inttostr(int i)
{
	std::stringstream ss;
	ss << i;
	std::string s(ss.str());
	return s;
}
/* Convert float to string */
std::string floattostr(float f)
{
	std::stringstream ss;
	ss << f;
	std::string s(ss.str());
	return s;
}

/***********************************/


/* Get dimensions of input file */

std::vector<int> get_dimensions(char *fileX)
{

	std::ifstream f(fileX);
	std::string line;

	int n = 0;
	
	while(getline(f,line))
	{
		n++;
	}

	std::vector<int> dim;
	dim.push_back(n);

	f.close();

	std::ifstream fd(fileX);
	std::string lined;

	int d = 0;
	
	while(getline(fd,lined))
	{
		std::vector<std::string> Xline = split(lined,',');
		d = Xline.size();
		break;
	}

	dim.push_back(d);

	return dim;
	
}

/***********************************/


/* Reads the data */
int read(float *x, float *y, char *file, int N, int D)
{

	std::ifstream f(file);
	std::string line;

	int i = 0;
	int n = 0;
	
	while(getline(f,line))
	{
		std::vector<std::string> Line = split(line,',');
		int d = 1;
		for(std::vector<std::string>::iterator it=Line.begin();it!=Line.end();++it)
		{
			if(d==1)
			{
				y[n] = strtofloat(*it);
			}
			else
			{	
				x[i] = strtofloat(*it);
				i++;
			}
			d++;
		}
			n++;
			
			if(d != D+2)
			{
				std::cout<<"There is missing data on the "<<n<<"th line!\n";
				return -1; 
			}
	}
	f.close();
	return 1;
}

/***********************************/

/* Checks for validity of flags' values */

int Check(int kernel_flag, float C, float Gamma, float epsilon)
{
	if((kernel_flag > 1) || (kernel_flag < 0) || (C < 0) || (Gamma < 0) || (epsilon < 0))
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

/***********************************/


/* Scales the data to feature-wise Mean = 0 and Stdev = 1*/

void Scale(float *Mean, float *Stdev,float *xTrain_scaled, float *xTrain, const int Ntrain, const int Dtrain)
{

	for(int d=0;d<Dtrain;d++)
	{
		Mean[d] = 0.0;
		for(int n=0;n<Ntrain;n++)
		{
			Mean[d] += xTrain[Dtrain*n+d];
		}
		Mean[d] = Mean[d]/(Ntrain*1.0);
	}

	for(int d=0;d<Dtrain;d++)
	{
		Stdev[d] = 0.0;
		for(int n=0;n<Ntrain;n++)
		{
			Stdev[d] += pow((xTrain[Dtrain*n+d]-Mean[d]),2.0); 
		}
		
		Stdev[d] = sqrt(Stdev[d]/((Ntrain-1)*1.0));
		/* We don't want to be dividing by 0 while normalizing */
		if(Stdev[d]<EPS)
		{
			Stdev[d] = 1;
		}
	}

	for(int i=0;i<Ntrain*Dtrain;i++)
	{	
		int d = (i%Dtrain);
		xTrain_scaled[i] = ((xTrain[i] - Mean[d])/Stdev[d]);
	}
}

/***********************************/


int main(int argc, char* argv[])
{

	/* Initialize variables whose values are provided by user through command line flags */
	char *filename = "train";

	int kernel_flag = 1;
	float C = 1;
	float Gamma = -1;
    float epsilon = 0.01;

	/* Parse command line flags */
	switch ( argc ) 
	{
		case 1:
			/* No training file specified */
			std::cout<<"Please specify training file!\n\nFormat is: mcmtrain train -Kernel 1 -C 100 -Gamma 0.1 -Epsilon 0.1\n(Please specify flags in this order only. If a flag is not specified, its default value is used.)\n\n";
			return 0;
			break;

		case 2:
			filename = argv[1]; 
			/* No flag specified, using default values */
			break;


		case 4:
			filename = argv[1];
			/* Kernel flag specified */
			kernel_flag = strtoint(argv[3]);
			if(Check(kernel_flag,1,1,1) != 1)
			{
				std::cout<<"Please provide valid values for the flags!\n-Kernel -> {0,1}, -C -> [0,+inf), -Gamma -> (0, +inf), -Epsilon -> (0, +inf)\n\n";
				return 0;
			}	

			break;

		case 6:
			filename = argv[1];
			/* Kernel_flag and C specified */
			kernel_flag = strtoint(argv[3]);
			C = strtofloat(argv[5]);
			if(Check(kernel_flag,C,1,1) != 1)
			{
				std::cout<<"Please provide valid values for the flags!\n-Kernel -> {0,1}, -C -> [0,+inf), -Gamma -> (0, +inf), -Epsilon -> (0, +inf)\n\n";
				return 0;
			}	
			break;

		case 8:
			filename = argv[1]; 
			/* Kernel flag, C and Gamma specified */
			kernel_flag = strtoint(argv[3]);
			C = strtofloat(argv[5]);
			Gamma = strtofloat(argv[7]);
			if(Check(kernel_flag,C,Gamma,1) != 1)
			{
				std::cout<<"Please provide valid values for the flags!\n-Kernel -> {0,1}, -C -> [0,+inf), -Gamma -> (0, +inf), -Epsilon -> (0, +inf)\n\n";
				return 0;
			}
        case 10:
			filename = argv[1]; 
			/* Kernel flag, C and Gamma specified */
			kernel_flag = strtoint(argv[3]);
			C = strtofloat(argv[5]);
			Gamma = strtofloat(argv[7]);
            epsilon = strtofloat(argv[9]);           
			if(Check(kernel_flag,C,Gamma,epsilon) != 1)
			{
				std::cout<<"Please provide valid values for the flags!\n-Kernel -> {0,1}, -C -> [0,+inf), -Gamma -> (0, +inf), -Epsilon -> (0, +inf)\n\n";
				return 0;
			}		

		break;

		default:
			std::cout<<"Incorrect format!\nFormat is: mcmtrain train -Kernel 1 -C 100 -Gamma 0.1 -Epsilon 0.1\n(Please specify flags in this order only. If a flag is not specified, its default value is used.)";
			return 0;
			break;
	}

	/* Initialize data storing variables, and populate them by reading data from training file */

	std::vector<int> dim = get_dimensions(filename);
	int Ntrain = dim[0];
	int Dtrain = dim[1]-1;

	float *xTrain = new float[Dtrain*Ntrain];
	float *xTrain_scaled = new float[Dtrain*Ntrain];
	float *Mean = new float[Dtrain];
	float *Stdev = new float[Dtrain];
	float *yTrain = new float[Ntrain];
	
	int read_flag = read(xTrain, yTrain, filename, Ntrain, Dtrain);
	if(read_flag < 1)
	{
		std::cout<<"Please check data format!\n\n";
		return 0;
	}

	/* Set default value of Gamma */
	if(argc<8)
	{
		Gamma = 1/(Dtrain*1.0); 
	}

	/* Scale features, before training */
	Scale(Mean, Stdev,xTrain_scaled,xTrain,Ntrain,Dtrain);

	/* Initialize variables to be used for solving the linear program */
	float *Vars = new float[3*Ntrain+3];


	/* Perform the training and time it */
	
	int begintr, endtr;
	begintr=clock();  
	trainMCM(Vars, xTrain_scaled, yTrain, Ntrain, Dtrain, C, kernel_flag, Gamma,epsilon);
	endtr=clock();
	std::cout << "\nTraining time for MCM is: " << double(diffclock(endtr,begintr)) << " s"<< std::endl;

	/* Compute results and store them, to be written to model file */

	/* Compute nSV */
	int nSV=0;
	for (int i=0;i<Ntrain;i++)
	{
		if (abs(Vars[i])>EPS)
		nSV++;
	}



	/* Output model file */

	/* Name of model file is derived from the first name of the train file: for eg: "xor.train" becomes "xor.model", or "xor.csv" becomes "xor.model" */
	std::string modelname = split(filename,'.')[0]+".model";
	std::string model = modelname;

	/*Write to model file*/
	std::ofstream mf(model.c_str());
	mf << "MCM Regression\n";
	if(kernel_flag == 0)
	{
		mf << "Kernel Linear(u*v')\n";
	}
	else
	{
		mf << "Kernel RBF(exp(-Gamma*(||u-v||^2)))\n";
	}
	mf << "C " << C << "\n";
	mf << "Gamma "<< Gamma << "\n";
    mf << "epsilon "<< epsilon << "\n";
	mf << "H " << Vars[3*Ntrain + 1] <<"\n";
	mf << "Bias " << Vars[Ntrain] <<"\n";
	mf << "P " << Vars[3*Ntrain+2] << "\n";
	for(int d=0;d<Dtrain-1;d++)
	{
		mf << Mean[d] <<",";
	}
	mf << Mean[Dtrain-1] <<"\n";

	for(int d=0;d<Dtrain-1;d++)
	{
		mf << Stdev[d] <<",";
	}
	mf << Stdev[Dtrain-1] <<"\n";

	mf << "nSV " << nSV << "\n";
	mf << "SV\n";
	for (int i=0;i<Ntrain;i++)
	{
		if (abs(Vars[i])>EPS)
		{
			mf << (Vars[i]);
			for (int d = 0;d<Dtrain;d++)
			{
				/* We write the scaled support vectors to the model file, since we are going to scale the test data also anyway */
				mf << "," << xTrain_scaled[i*Dtrain+d];
			}
			mf << "\n";
		}
	}

	/* Free up space */

	delete [] xTrain;
	delete [] yTrain;
	delete [] xTrain_scaled;
	delete [] Mean;
	delete [] Stdev;
	delete [] Vars;

	return 0;
}

