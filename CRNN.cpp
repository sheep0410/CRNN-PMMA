#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#define N 1000
#define hcn 5
#define san 5
#define van 5
#define h1n 6
#define h2n 6
#define h3n 6
#define inputNo 4
#define hiddenNo 3
#define outNo 2
#pragma warning(disable:4996)
double hc[hcn * san * van * h1n * h2n * h3n];
double sa[hcn * san * van * h1n * h2n * h3n];
double va[hcn * san * van * h1n * h2n * h3n];
double h1[hcn * san * van * h1n * h2n * h3n], h2[hcn * san * van * h1n * h2n * h3n], h3[hcn * san * van * h1n * h2n * h3n];
double h10, h20, h30;
double delta_z = 20e-3 / N;
double delta_t = 1e-1;
double emission = 1;
double Tn[18000], rho_PMMA[18000], rho_MMA[18000], rho_gasn[18000], Qn[18000];
double Tn1[18000], rho_PMMA1[18000], rho_MMA1[18000], rho_gasn1[18000], Qn1[18000];
double r2w31[18000], r3w31[18000], r2w41[18000], r3w41[18000], r2b1[18000], r3b1[18000];
double gw31, gw41, gwb1, gw32, gw33, gw42, gw43, gwb2, gwb3;
double wgn[18000], wdn[18000], heatflux[18000];
double surfT[18000], mass[18000];
double massloss[18000];
double mln[18000];
double A[N + 2][N + 2];
double B[N + 2];
double Hs1 = 540e3;
double rho_pmma = 1187.8;
double A1 = 0, E1 = 1e5, R = 8.314e-3, n1 = 1;
double absorb = 1.3786e3;
double surfaceabsorb = 0;
double sigma = 5.67e-8;
double hconv = 15;
double endtime = 0;
double xlambda, xbeta, xspecific, xconductivity, xdensity, xhconv, xsurf, xdepth;
double x1[N + 2], x2[N + 2], x3[N + 2], x4[N + 2], r1[N + 2], r2[N + 2], r3[N + 2];
double ml[N + 2];

class Neuralnetwork
{
public:
	double win[inputNo][hiddenNo];
	double hiddenInput[hiddenNo];
	double hiddenOutput[hiddenNo];
	double lnA[hiddenNo];
	Neuralnetwork()
	{
		for (int i = 0; i < hiddenNo; i++)
		{
			hiddenInput[i] = 0;
			hiddenOutput[i] = 0;
			lnA[i] = 0;
			for (int j = 0; j < inputNo; j++)
				win[j][i] = 0;
		}
		win[0][0] = 1; win[0][1] = 0; win[0][2] = 1;
		win[1][0] = 0; win[1][1] = 1; win[1][2] = 0;
		win[2][0] = 0.0; win[2][1] = 0.1; win[2][2] = 0.1;
		win[3][0] = 1.35e2; win[3][1] = 1.05e2; win[3][2] = 1.25e2;

		lnA[0] = 18;    lnA[1] = 15;    lnA[2] = 18;
	};
	void Copy(Neuralnetwork nn)
	{
		for (int i = 0; i < hiddenNo; i++)
		{
			lnA[i] = nn.lnA[i];
			for (int j = 0; j < inputNo; j++)
				win[j][i] = nn.win[j][i];
		}
	}
	void Predict(double* input, double* output)
	{
		for (int i = 0; i < hiddenNo; i++)
		{
			hiddenInput[i] = 0;
			for (int j = 0; j < inputNo; j++)
			{
				hiddenInput[i] += input[j] * win[j][i];
			}
			hiddenOutput[i] = exp(hiddenInput[i] + lnA[i]);
		}
		output[0] = hiddenOutput[0] - hiddenOutput[1];
		output[1] = hiddenOutput[1] + hiddenOutput[2];
	}
	void print()
	{
		printf("win: \n");
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				printf(" % f     ", win[i][j]);
			}
			printf("\n");
		}
		printf("lnA: \n");
		printf("%f  %f  %f", lnA[0], lnA[1], lnA[2]);
	}
	~Neuralnetwork()
	{};
};
Neuralnetwork nn;
Neuralnetwork minNN;
double mabs(double x)
{
	return x < 0 ? -x : x;
}
int MatrixSolve(double* a, double* b, int n)
{
	int i, j, k, * js, is;
	double max, t;
	for (i = 1; i < n; i++)
	{
		B[i] = B[i] - B[i - 1] * A[i][i - 1] / A[i - 1][i - 1];
		A[i][i] = A[i][i] - A[i - 1][i] * A[i][i - 1] / A[i - 1][i - 1];
		A[i][i - 1] = 0;
		B[i] = B[i] / A[i][i];
		if (i < n - 1)
			A[i][i + 1] = A[i][i + 1] / A[i][i];
		A[i][i - 1] = A[i][i - 1] / A[i][i];
		A[i][i] = 1;

	}
	B[n - 1] = B[n - 1] / A[n - 1][n - 1];
	for (i = n - 2; i >= 0; i--)
	{
		B[i] = (B[i] - A[i][i + 1] * B[i + 1]) / A[i][i];
	}
	return(1);
}

void Initialization()
{
	char heatfile[200];
	for (int ii = 0; ii < N + 2; ii++)
	{
		Tn[ii] = 300; rho_PMMA[ii] = rho_pmma; rho_gasn[ii] = 0; Qn[ii] = 0;
		rho_MMA[ii] = 0; r2w31[ii] = 0; r3w31[ii] = 0; r2w41[ii] = 0; r3w41[ii] = 0;
		r2b1[ii] = 0; r3b1[ii] = 0;
		Tn1[ii] = 300; rho_PMMA1[ii] = rho_pmma; rho_gasn1[ii] = 0; Qn1[ii] = 0;
		rho_MMA1[ii] = 0;
		wgn[ii] = 0; wdn[ii] = 0;
	}
	for (int ii = 0; ii < endtime; ii++)
	{
		surfT[ii] = 0;
		mass[ii] = 0;
	}
	gw31 = 0; gw32 = 0; gw33 = 0;
	gw41 = 0; gw42 = 0; gw43 = 0;
	gwb1 = 0; gwb2 = 0; gwb3 = 0;
}
double max(double x1, double x2)
{
	return x1 > x2 ? x1 : x2;
}
double min(double x1, double x2)
{
	return x1 < x2 ? x1 : x2;
}
double sgn(double x1)
{
	return x1 > 0 ? 1 : -1;
}
void Reaction(double time)
{
	int ii;
	int time_index = round(time / delta_t);
	double input[4];
	double output[2];
	double r2w31n = 0, r3w31n = 0, r2w41n = 0, r3w41n = 0, r2b1n = 0, r3b1n = 0;
	double x3r2 = 0, x3r3 = 0, x4r2 = 0, x4r3 = 0, rr2 = 0, rr3 = 0;
	mln[time_index] = 0;
	if (time > 140.45)
		time = time;
	for (ii = 0; ii < N + 2; ii++)
	{
		input[0] = log(max(rho_PMMA[ii], 1e-18));
		input[1] = log(max(rho_MMA[ii], 1e-18));
		input[2] = log(Tn[ii]);
		input[3] = -1 / Tn[ii] / R;
		nn.Predict(input, output);
		rho_PMMA[ii] = max(rho_PMMA[ii] - nn.hiddenOutput[0] - nn.hiddenOutput[2], 1e-18);
		rho_MMA[ii] = max(rho_MMA[ii] + nn.hiddenOutput[0] - nn.hiddenOutput[1], 1e-18);
		if (ii == 0 || ii == N + 1)
			continue;
		x1[ii] = input[0];    x2[ii] = input[1];
		x3[ii] = input[2];    x4[ii] = input[3];
		r1[ii] = nn.hiddenOutput[0];
		r2[ii] = nn.hiddenOutput[1];
		r3[ii] = nn.hiddenOutput[2];
		ml[ii] = r2[ii] + r3[ii];
		if (r2[ii] > 1e-10)
			r2w31[ii] = r2w31[ii] + r2[ii] / exp(x2[ii]) * (r1[ii] * x3[ii] - r2w31[ii]);
		if (r3[ii] > 1e-10)
			r3w31[ii] = r3w31[ii] + r3[ii] / exp(x1[ii]) * (-r1[ii] * x3[ii] - r3w31[ii]);
		if (r2[ii] > 1e-10)
			r2w41[ii] = r2w41[ii] + r2[ii] / exp(x2[ii]) * (r1[ii] * x4[ii] - r2w41[ii]);
		if (r3[ii] > 1e-10)
			r3w41[ii] = r3w41[ii] + r3[ii] / exp(x1[ii]) * (-r1[ii] * x4[ii] - r3w41[ii]);
		if (r2[ii] > 1e-10)
			r2b1[ii] = r2b1[ii] + r2[ii] / exp(x2[ii]) * (r1[ii] - r2b1[ii]);
		if (r3[ii] > 1e-10)
			r3b1[ii] = r3b1[ii] + r3[ii] / exp(x1[ii]) * (-r1[ii] - r3b1[ii]);
		r2w31n += r2w31[ii] * delta_z; r3w31n += r3w31[ii] * delta_z; r2w41n += r2w41[ii] * delta_z;
		r3w41n += r3w41[ii] * delta_z; r2b1n += r2b1[ii] * delta_z; r3b1n += r3b1[ii] * delta_z;
		x3r2 += x3[ii] * r2[ii]; x3r3 += x3[ii] * r3[ii]; x4r2 += x4[ii] * r2[ii];
		x4r3 += x4[ii] * r3[ii]; rr2 += r2[ii]; rr3 += r3[ii];
		mln[time_index] += ml[ii];
	}
	gw31 = gw31 + (2 * (-massloss[time_index] + mln[time_index])) * (r2w31n + r3w31n) * delta_t;
	if (_isnan(gw31))
		gw31 = gw31;
	gw41 = gw41 + (2 * (-massloss[time_index] + mln[time_index])) * (r2w41n + r3w41n) * delta_t;
	gwb1 = gwb1 + (2 * (-massloss[time_index] + mln[time_index])) * (r2b1n + r3b1n) * delta_t;
	gw32 = gw32 + (2 * (-massloss[time_index] + mln[time_index]) * x3r2) * delta_t;
	gw33 = gw33 + (2 * (-massloss[time_index] + mln[time_index]) * x3r3) * delta_t;
	gw42 = gw42 + (2 * (-massloss[time_index] + mln[time_index]) * x4r2) * delta_t;
	gw43 = gw43 + (2 * (-massloss[time_index] + mln[time_index]) * x4r3) * delta_t;
	gwb2 = gwb2 + (2 * (-massloss[time_index] + mln[time_index]) * rr2) * delta_t;
	gwb3 = gwb3 + (2 * (-massloss[time_index] + mln[time_index]) * rr3) * delta_t;
}
double q(double x)
{
	return x * 350 * 0.95;
}
void TemperatureEvolution(double t, double hfx)
{
	double boundary0;
	double specific[N + 2], conductivity[N + 2];
	double sourceterm = 0;
	double qq = q(t);

	int time_index = round(t / delta_t);
	double attentuation = 1;
	double msa = 0;
	boundary0 = surfaceabsorb * qq * attentuation - 0.95 * sigma * (pow(Tn[1], 4) - pow(300, 4)) - hconv * (Tn[1] - 300);
	memset(A, 0, (N + 2) * (N + 2) * sizeof(double));

	for (int ii = 0; ii < N + 2; ii++)
	{
		if (Tn[ii] < 378)
		{
			specific[ii] = 1505 * pow(Tn[ii] / 300, 0.89);// (173 + 204.1e3 / Tn[ii] / Tn[ii] + 4.341 * Tn[ii]);
			conductivity[ii] = 0.2 * pow(Tn[ii] / 300, -0.19);;
		}
		else
		{
			specific[ii] = 1505 * pow(Tn[ii] / 300, 0.89);// (173 + 204.1e3 / Tn[ii] / Tn[ii] + 4.341 * Tn[ii]);
			conductivity[ii] = 0.2 * pow(Tn[ii] / 300, -0.19);;
		}
	}
	double TE0 = Tn[0] + boundary0 * delta_z / conductivity[0];
	double TEN = Tn[N - 1];
	double Con0 = 0.21;
	double ConN = 0.21;
	double temp1, temp2, temp3, temp4;
	A[0][0] = 1; A[0][1] = -1; B[0] = boundary0 * delta_z / conductivity[0];
	for (int ii = 1; ii <= N; ii++)
	{
		temp1 = conductivity[ii] * (Tn[ii + 1] - 2 * Tn[ii] + Tn[ii - 1]) / delta_z / delta_z;
		if (mabs((double)(Tn[ii + 1] - 2 * Tn[ii] + Tn[ii - 1])) > 100)
			ii = ii;
		temp3 = (1 - surfaceabsorb) * qq * absorb * exp(-absorb * (ii - 1) * delta_z);
		temp4 = -ml[ii] * Hs1;
		double source = temp3 + temp4 + Tn[ii] * (rho_PMMA[ii] + rho_MMA[ii]) * specific[ii] / delta_t;
		if (rho_PMMA[ii] + rho_MMA[ii] < 1e-6)
		{
			ii = ii;
		}
		A[ii][ii - 1] = -conductivity[ii] / delta_z / delta_z;
		A[ii][ii] = (rho_PMMA[ii] + rho_MMA[ii]) * specific[ii] / delta_t + 2 * conductivity[ii] / delta_z / delta_z;
		A[ii][ii + 1] = -conductivity[ii] / delta_z / delta_z;
		B[ii] = source;
		A[ii][ii - 1] = A[ii][ii - 1] / A[ii][ii];
		A[ii][ii + 1] = A[ii][ii + 1] / A[ii][ii];
		B[ii] = B[ii] / A[ii][ii];
		A[ii][ii] = 1;
	}
	A[N + 1][N] = -1; A[N + 1][N + 1] = 1; B[N + 1] = 0;
	MatrixSolve(*A, B, N + 2);

	if (Tn[1] > B[1])
		Con0 = Con0;
	for (int ii = 0; ii < N + 2; ii++)
	{
		Tn1[ii] = B[ii];
		Tn[ii] = Tn1[ii];
		if (_isnan(Tn[ii]))
			ii = ii;
	}
	if (Tn[1] > Tn[0])
		Tn[0] = Tn[0];

}
void Deinitialization()
{
}
void main()
{
	double stopTime[23];
	double minDev = 1e6;
	int timestep = 1;
	double t;
	double rA1, rE1, rabsorb, rsurfaceabsorb;
	double hc_min = 8, hc_max = 100;
	double va_min = 200, va_max = 2000, sa_min = 0, sa_max = 1;
	double h1_min = 0, h1_max = 60, h2_min = 0, h2_max = 60, h3_min = 0, h3_max = 60, h4_min = 0, h4_max = 50;
	double lambda_buf[9] = {
		1.00,    1.6,  2.5,    4,    6.3,   10,   15.9,    25.2,     40
	};

	char filename[100];
	endtime = 1800;
	double hfx = 40000 * 0.945;
	double aaa, bbb, ccc, ddd, eee, fff;
	fpignition = fopen("G:\\ignitiontime.txt", "w+");
	FILE* fptemp;
	fptemp = fopen("G:\\temperature.txt", "w+");
	FILE* fptempdis;
	fptempdis = fopen("G:\\temperatureDis.txt", "w+");
	FILE* finput;
	finput = fopen("G:\\inputvector.txt", "w+"); */
		double gradient[18000];
	double massDev[18000];
	double minMassloss[18000];
	rho_pmma = 1150;
	hconv = 10;
	FILE* fmass;
	fmass = fopen("C:\\Users\\95295\\Desktop\\CINN\\PMMA\\mass loss rate.txt", "r");
	double t0, t1; int nR = 0;
	int i;
	for (i = 0; i < endtime / delta_t; i++)
	{
		nR = fscanf(fmass, "%lf %lf", &t0, &(massloss[i]));
		if (nR == -1)
			break;
	}
	endtime = (i - 1) * delta_t;
	fclose(fmass);
	nn.win[0][0] = 1; nn.win[0][1] = 0; nn.win[0][2] = 1;
	nn.win[1][0] = 0; nn.win[1][1] = 1; nn.win[1][2] = 0;
	nn.win[2][0] = 0.05; nn.win[2][1] = 0.05; nn.win[2][2] = 0.05;
	nn.win[3][0] = 1.36e2; nn.win[3][1] = 1.06e2; nn.win[3][2] = 1.26e2;
	nn.lnA[0] = 20.9;    nn.lnA[1] = 19.9;    nn.lnA[2] = 20.4;
	int nOption = 300;
	for (int option = 0; option < nOption; option++)
	{
		printf("option = %d\n", option);
		gw31 = 0; gw32 = 0; gw33 = 0;
		gw41 = 0; gw42 = 0; gw43 = 0;
		gwb1 = 0; gwb2 = 0; gwb3 = 0;
		absorb = 2000;
		timestep = 1;
		Initialization();
		mass[0] = 0;
		double totalmass = 0;
		for (int jj = 1; jj < N + 1; jj++)
		{
			mass[0] = mass[0] + (rho_PMMA[jj] + rho_MMA[jj]) * delta_z;
		}

		for (t = delta_t; t <= endtime; t = t + delta_t)
		{
			Reaction(t);
			TemperatureEvolution(t, hfx);
			if (t > 150)
				t = t;
			surfT[timestep] = Tn[1];
			mass[timestep] = 0;
			for (int jj = 0; jj < N; jj++)
			{
				mass[timestep] = mass[timestep] + (rho_PMMA[jj] + rho_MMA[jj]) * delta_z;
			}
			timestep++;
		}

		sprintf(filename, "C:\\Users\\95295\\Desktop\\CINN\\PMMA\\masslosstime%d.txt", option);
		fmass = fopen(filename, "w+");
		for (int nt = 1; nt <= round(endtime / delta_t + 1e-6); nt = nt + 1)
			fprintf(fmass, "%f %f %f\n", nt * 0.1, surfT[nt], mln[nt]);
		fclose(fmass);
		double dev = 0; double sumM = 0;
		for (int i = 1; i <= endtime / delta_t; i++)
		{
			dev = dev + (mln[i] - massloss[i]) * (mln[i] - massloss[i]) * delta_t;
		}
		if (dev < minDev)
		{
			minDev = dev;
			minNN.Copy(nn);
			for (int nt = 0; nt <= round(endtime / delta_t) + 1e-6; nt = nt + 1)
				minMassloss[nt] = mln[nt];
		}
		double ampl = 0.1;
		if (option > 0)
		{
			if (gradient[option - 1] > 0.05)
				ampl = 0.01;
			else
			{
				ampl = max(0.2 * gradient[option - 1] / 1, 0.001);
			}
		}
		double eta = rand() * ampl / RAND_MAX; double thresh = 0.01;
		double frand = rand() * 0.01 / RAND_MAX - 0.005;
		nn.win[2][0] = nn.win[2][0] - sgn(gw31) * min(eta * mabs(gw31), mabs(nn.win[2][0]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.win[2][1] = nn.win[2][1] - sgn(gw32) * min(eta * mabs(gw32), mabs(nn.win[2][1]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.win[2][2] = nn.win[2][2] - sgn(gw33) * min(eta * mabs(gw33), mabs(nn.win[2][2]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.win[3][0] = nn.win[3][0] - sgn(gw41) * min(eta * mabs(gw41), mabs(nn.win[3][0]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.win[3][1] = nn.win[3][1] - sgn(gw42) * min(eta * mabs(gw42), mabs(nn.win[3][1]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.win[3][2] = nn.win[3][2] - sgn(gw43) * min(eta * mabs(gw43), mabs(nn.win[3][2]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.lnA[0] = nn.lnA[0] - sgn(gwb1) * min(eta * mabs(gwb1), mabs(nn.lnA[0]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.lnA[1] = nn.lnA[1] - sgn(gwb2) * min(eta * mabs(gwb2), mabs(nn.lnA[1]) * thresh) + frand;
		eta = rand() * ampl / RAND_MAX; frand = rand() * 0.001 / RAND_MAX - 0.0005;
		nn.lnA[2] = nn.lnA[2] - sgn(gwb3) * min(eta * mabs(gwb3), mabs(nn.lnA[2]) * thresh) + frand;
		Deinitialization();
		//printf("gw20 = %f, gw21 = %f, gw22 = %f\n", gw31, gw32, gw33);
		//printf("gw30 = %f, gw31 = %f, gw32 = %f\n", gw41, gw42, gw43);
		//printf("b1 = %f, b2 = %f, b3 = %f\n", gwb1, gwb2, gwb3);

		massDev[option] = dev;
		gradient[option] = sqrt(gw31 * gw31 + gw32 * gw32 + gw33 * gw33
			+ gw41 * gw41 + gw42 * gw42 + gw43 * gw43
			+ gwb1 * gwb1 + gwb2 * gwb2 + gwb3 * gwb3);
		printf("   dev = %f, gradient = %f\n", dev, gradient[option]);
		if (gradient[option] < 1.5e-4)
			break;
	}

	fmass = fopen("C:\\Users\\95295\\Desktop\\CINN\\PMMA\\massgrad.txt", "w+");
	for (int option = 0; option < nOption; option++)
	{
		fprintf(fmass, "%d %f %f\n", option, massDev[option], gradient[option]);
	}
	fclose(fmass);
	fmass = fopen("C:\\Users\\95295\\Desktop\\CINN\\PMMA\\minmassloss.txt", "w+");
	for (int nt = 1; nt <= round(endtime / delta_t) + 1e-6; nt = nt + 1)
		fprintf(fmass, "%f %f %f\n", nt * 0.1, surfT[nt], minMassloss[nt]);
	fclose(fmass);
	printf("minDev: %f\n", minDev);
	minNN.print();
}