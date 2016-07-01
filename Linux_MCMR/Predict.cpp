/* This file contains code for mcmpredict. It takes as input the test file and the model file and produces the output file. */

#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include "string.h"
#include "soplex-1.7.2/src/soplex.h"

using namespace soplex;

#define EPS 1/infinity 
#define INF infinity 

#define C_val 0.0001
#define Beta 1


/* This is an auxiliary function for timekeeping */
double diffclock(int clock1,int clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(float(diffticks))/CLOCKS_PER_SEC;
	return diffms;
}


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
		std::cout<<"Improper choice for kernel parsed from model file!\n";
		return 0;
	}		
}

/***********************************/


/* Auxiliary string functions */

/* Splitting a string by delimiter */
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


/* Reads the model file */
int read_model(char *model_file, int Dtest,int &kernel_flag, float &Gamma, float &P, float *Mean, float *Stdev, float *lambda, float *SV)
{
	std::ifstream mf(model_file);
	std::string line;

	int i = 0;
	int d = 0;
	int pos = 0;
	int l = 0;
	
	while(getline(mf,line))
	{
		l++;
		if(l == 2)/* kernel_flag */
		{
			std::vector<std::string> Line = split(line,' ');
			if(Line[1][0] == 'R')
			{
				kernel_flag = 1;
			}
			else
			{
				kernel_flag = 0;
			}
		}
		else if(l==4)/* Gamma */
		{
			std::vector<std::string> Line = split(line,' ');
			Gamma = strtofloat(Line[1]);
		}
		else if(l==8)/* P */
		{
			std::vector<std::string> Line = split(line,' ');
			P = strtofloat(Line[1]);
		}
		else if(l==9)/* Mean */
		{
			std::vector<std::string> Line = split(line,',');
			int d = 0;
			for(std::vector<std::string>::iterator it=Line.begin();it!=Line.end();++it)
			{
				Mean[d] = strtofloat(*it);
				d++;
			}
			if(Dtest != d)
			{
				std::cout<<"Mean data has inconsistent dimensions!\n";
				return -1;
			}
		}
		else if(l==10)/* Stdev */
		{
			std::vector<std::string> Line = split(line,',');
			int d = 0;
			for(std::vector<std::string>::iterator it=Line.begin();it!=Line.end();++it)
			{
				Stdev[d] = strtofloat(*it);
				d++;
			}
			if(Dtest != d)
			{
				std::cout<<"Stdev data has inconsistent dimensions!\n";
				return -1;
			}
			
		}
		if(l<=12)
		{
			continue;
		}

		std::vector<std::string> Line = split(line,',');
		int pos = 0;
		int d = 0;
		for(std::vector<std::string>::iterator it=Line.begin();it!=Line.end();++it)
		{
			pos++;
			if(pos==1)
			{
				lambda[i] = strtofloat(*it);
			}
			else
			{
				SV[Dtest*i+d] = strtofloat(*it);
				d++;
			}
		}
		if(Dtest != d)
		{
			std::cout<<"Training data and Test data have inconsistent dimensions at line "<<i+1<<" d is: "<<d<<" Dtest is: "<<Dtest<<"!\n";
			return -1;
		}
		i++;
	}
	mf.close();
	return 1;
}

/***********************************/


/* Obtains the bias and nSV value from the model file */

std::vector<std::string> get_bias_nsv(char *model_file)
{
	std::ifstream mf(model_file);
	std::string line;

	std::vector<std::string> bias_nsv;

	int l = 0;
	
	while(getline(mf,line))
	{
		l++;
		if(l>11)
		{
			break;
		}
		else if(l == 7)/* line for bias */
		{
			bias_nsv.push_back(split(line,' ')[1]);
		}
		else if(l == 11)/* line for nsv */
		{
			bias_nsv.push_back(split(line,' ')[1]);
		}
	}
	
	return bias_nsv;	
}

/***********************************/


/* This function performs the prediction */

float predict(float *PredictedLabels,int kernel_flag,  float Gamma, const float P, const float *SV, const float *lambda, const int nSV, const float bias, const float* xTest, const float *yTest, const int Ntest, const int D)
{
	/* The predicted value for point i is: (-1/P) * Sum_i( Sum_j( (lambda_j*K_i_j) + bias) ) */

	float MSE = 0.0;

	for (int n1=0;n1 < Ntest; n1++)
	{
		float s = bias;
		for (int n2 = 0; n2 < nSV; n2++)
		{
			s += (lambda[n2]) * kernel(kernel_flag, Gamma, xTest+n1*D, SV+n2*D, D); /* Adds lamba_j*K_i_j */
		}

		s = (((-1.0)/P) * s);
		PredictedLabels[n1] = s;

		MSE += (yTest[n1] - s)*(yTest[n1] - s);
}
	MSE = (MSE / (Ntest*1.0));	

	return MSE;
}

/***********************************/


/* This scales the test data according to the Mean and Std. Deviation of the Train data, which is obtained from the model file */

void Scale(float *xTest_scaled, float *xTest, float *Mean, float *Stdev, const int Ntest, const int Dtest)
{
	for(int i=0;i<Ntest*Dtest;i++)
	{
		int d = i%Dtest;
		xTest_scaled[i] = ((xTest[i] - Mean[d])/Stdev[d]);
	}
}

/***********************************/

/* This calculates the Pearson correlation and Fisher's Z score */

float Pearson(float *yTest, float *PredictedLabels, int Ntest)
{
    int i;
    float sx=0,sy=0,sxy=0,sxx=0,syy=0,mx,my,sdx,sdy,cxy,r,vx,vy;
    
    for(i=0;i<Ntest;i++)
    {
        sx+=yTest[i];
        sxx+=(yTest[i]*yTest[i]);
        
        sy+=PredictedLabels[i];
        syy+=(PredictedLabels[i]*PredictedLabels[i]);
        sxy+=(yTest[i]*PredictedLabels[i]);
    }
    mx=sx/Ntest;
    my=sy/Ntest;
    vx=(sxx/Ntest)-(mx*mx);
    vy=(syy/Ntest)-(my*my);
    sdx=sqrt(vx);
    sdy=sqrt(vy);
    cxy=(sxy/Ntest)-(mx*my);
    r=cxy/(sdx*sdy);
    std::cout << "Pearson correlation coefficient : " << r << std::endl;
    std::cout << "Fisher Z is "<< atanh(r) << std::endl;
    return r;
} 

int main(int argc, char* argv[])
{

	/* Initialize file names, to be parsed as command line flags */
	char *test_file = "test";
	char *model_file = "model";

	/* Parse command line flags */
	switch ( argc ) 
	{
		case 1:
			/* No testing file specified */
			std::cout<<"Please specify test and model file!\n\nFormat is: mcmpredict test_file model_file\n\n";
			return 0;
			break;

		case 2:
			/* No model file specified */
			std::cout<<"Please specify model file!\n\nFormat is: mcmpredict test_file model_file\n\n";
			return 0;
			break;

		case 3:
			/* Correct format */
			test_file = argv[1];
			model_file = argv[2];
			break;		

		default:
			std::cout<<"Incorrect format!\nFormat is: mcmpredict test_file model_file\n\n";
			return 0;
			break;
	}

	/* We now initialize data variables and populate them by reading the test file and the model file */
	std::vector<int> dim = get_dimensions(test_file);
	int Ntest = dim[0];
	int Dtest = dim[1]-1;

	std::vector<std::string> bias_nsv = get_bias_nsv(model_file);
	float bias = strtofloat(bias_nsv[0]);
	int nSV = strtoint(bias_nsv[1]);

	float *xTest = new float[Dtest*Ntest];
	float *xTest_scaled = new float[Dtest*Ntest];
	float *yTest = new float[Ntest];
	float *Mean = new float[Dtest];
	float *Stdev = new float[Dtest];
	float *SV = new float[Dtest*nSV];
	float *lambda = new float[nSV];
	float P = 1.0;

	/* Initialize with the default values for the flags */
	int kernel_flag = 1;
	float Gamma = 1/(Dtest*1.0);

	int read_flag = read(xTest, yTest, test_file, Ntest, Dtest);
	if(read_flag < 1)
	{
		std::cout<<"Please check data format!\n\n";
		return 0;
	}

	read_flag = read_model(model_file,Dtest,kernel_flag, Gamma, P, Mean,Stdev,lambda,SV);

	
	if(read_flag < 1)
	{
		std::cout<<"Please check data format!\n\n";
		return 0;
	}

	/* Initialize variable to store the predicted labels */	
	float *PredictedLabels = new float[Ntest];

	/* This scales the test data as per the mean and stdev of the test data, obtained from the model file */
	Scale(xTest_scaled, xTest, Mean, Stdev,Ntest,Dtest);

	/* We now perform the prediction and also time it*/
	int begintr, endtr;

	begintr=clock();  
	float results = predict(PredictedLabels, kernel_flag, Gamma, P, SV, lambda, nSV, bias, xTest_scaled, yTest, Ntest, Dtest);
	endtr=clock();
	std::cout << "\nTesting(Prediction) time for MCM is: " << double(diffclock(endtr,begintr)) << " s\n"<< std::endl;

	float r = Pearson(yTest, PredictedLabels, Ntest);

	/* Create Output file. Output file's name is derived from the name of the test file: "xor.test" becomes "xor.output", or "xor.csv" becomes "xor.output" */
	std::string outputname = split(test_file,'.')[0]+".output";
	std::string output = outputname;
    std::string plotname = split(test_file,'.')[0]+".plot";
	std::string plot = plotname;

	/* Write to the output file*/
	std::ofstream Of(output.c_str());
    std::ofstream Op(plot.c_str());
    
	Of << "nSV is "<<nSV<<"\n";	
	Of << "MSE is "<<results<<"\n";
    Of << "Pearson R is "<<r<<"\n";
    Of << "Fisher Z is "<<atanh(r)<<"\n";

	/* Write predicted labels to the output file */
	for(int i=0;i<Ntest;i++)
	{
		Of << PredictedLabels[i] << "\n";
        Op << i+1 <<" " << yTest[i] << " " << PredictedLabels[i] << std::endl;
	}

	/* Free up space */ 
	delete [] xTest;
	delete [] yTest;
	delete [] xTest_scaled;
	delete [] Mean;
	delete [] Stdev;
	delete [] SV;
	delete [] lambda;
	
	return 0;
}

