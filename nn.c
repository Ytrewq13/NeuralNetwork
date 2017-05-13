#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <malloc.h>

#define IN_SIZE (3) // The size of the input vectors.
#define OUT_SIZE (3) // The size of the output vectors.

uint max(uint *arr, uint length)
{
	/* A function to find the largest int in an array of ints.*/
	uint i;
	uint record = arr[0];
	for (i = 0; i < length; i++)
	{
		if (arr[i] > record)
		{
			record = arr[i];
		}
	}
	return record;
}

double * matMultVect(double *vector, uint *mLen, double matrix[mLen[0]][mLen[1]], uint vLen)
{
	assert(vLen == mLen[1]);
	uint i, j;
	static double *result;
	result = malloc(mLen[0] * sizeof(result));
	for (i = 0; i < mLen[0]; i++)
	{
		for (j = 0; j < mLen[1]; j++)
		{
			printf("[%d,%d]\n", i, j);
			result[i] += matrix[i][j] * vector[j];
		}
	}
	return result;
}

double * matMult
(uint *len1, uint *len2, double mat1[len1[0]][len1[1]], double mat2[len2[0]][len2[1]])
{
	assert(len1[1] == len2[0]);
	uint i, j, k;
	double *result;
	result = malloc(len1[0] * len2[1] * sizeof(result));
	for (i = 0; i < len1[0]; i++)
	{
		for (j = 0; j < len2[1]; j++)
		{
			for (k = 0; k < len1[1]; k++)
			{
				*(result + i + (j * len1[0])) += mat1[i][k] * mat2[k][j];
				printf("%u, %u, %u\n", i,j,k);
				printf("%10.0f\n", *(result + j + (i * len1[0])));
			}
			printf("%20.0f\n", *(result + j + (i * len1[0])));
		}
	}
	return result;
}

double sigmoid(double value, bool derivative)
{
	/* A function to calculate and return the sigmoid activation function.*/
	if (derivative)
	{
		// Calculate the derivative of the value passed.
		assert(value <= 1);
		assert(value >= 0);
		double sig = sigmoid(value, false);
		return sig * (1 - sig);
	} else {
		return (1 / (1 + exp(-1 * value)));
	}
}



void feedForward
(double *input, double *weights, double *outputs, uint *topology, uint lNum)
{
	/* input : a pointer to the input vector.
	 * weights : a pointer to the synapse weights.
	 * outputs : a pointer to the array in which to store calculated outputs.
	 * topology : a pointer to the layer sizes array.
	 * lNum : the number of layers. Will be used to access the topology array,
	 *        which will then be used to calculate the size of each weights array
	 *        and each outputs array. These calculations will be the most likely
	 *        cause of segmentation fault (core dumped) errors.
	 */
	// Variable declarations.
	//
	uint inSize = topology[0]; // Size of the input array.
	uint outSize = topology[lNum];
	uint weightDims[3]; // The dimensions of the weights array.
	uint * neuralOutputs;
	neuralOutputs = malloc(sizeof(neuralOutputs) * lNum * max(topology, lNum));
	// Calculate the size of each array.
	//
	weightDims[0] = lNum - 1;
	weightDims[1] = max(topology, lNum);
	weightDims[2] = max(topology, lNum);
	// Feed the input through the network.
	//
	double *w = weights;
	uint i, j, k;
	for (i = 0; i < lNum; i++)
	{
		w += topology[i] * topology[i+1];
		double wm[topology[i+1]][topology[i]];
		neuralOutputs[i] = matMult();
	}
}

int main(int argc, char **argv)
{
	// variable declarations.
	//
	uint i, j, k;
	uint layerCount = 3; // Number of layers, including input and output.
	uint layerSizes[] = {IN_SIZE, 4, OUT_SIZE}; // The size of the neuron layers.
	uint *p_layerSizes = (uint*)layerSizes;
	uint lSizeMax = max(layerSizes, layerCount);
	double *p_input = malloc(layerSizes[0] * sizeof(p_input));
	double weights[layerCount-1][lSizeMax][lSizeMax];
	double *p_weights = (double*)weights; /* This method of setting up
	 * the weights array means that some memory will be wasted, but it will be
	 * possible to change how many layers there are without having to use a single
	 * one-dimensional array to store all the weights, which would require very
	 * annoying and confusing maths to calculate an index.*/
	double deltas[layerCount-1][lSizeMax][lSizeMax];
	double *p_deltas = (double*)deltas; /* This array will store the
	 * change to be added to each weight. These values are calculated during
	 * backpropagation.*/
	double outputs[layerCount][lSizeMax];
	double *p_output = (double*)outputs; /* Array to store the outputs of each
	 * layer of the net. These will be the values output by the neurons and will
	 * be used for backpropagation. Each layer of outputs will be a n*1 vector,
	 * so a two-dimensional array will be sufficient for this purpose.*/
	printf("The weights are stored in a %dx%dx%d array.\n", layerCount,lSizeMax, lSizeMax);
	// Randomise the weights.
	//
	srand(time(NULL));
	for (i = 0; i < layerCount-1; i++)
	{
		for (j = 0; j < layerSizes[i+1]; j++)
		{
			for (k = 0; k < layerSizes[i]; k++)
			{
				// Randomise element weights[i][j][k] between -1 and 1.
				double value = (double)rand() / ((double)RAND_MAX / 2.0) - 1.0;
				weights[i][j][k] = value;
				//printf("%f\n", value);
				// This works.
			}
		}
	}
	printf("The weights are randomised.\n");
	printf("The sigmoid of 0 is %f\n", sigmoid(0,false));
	printf("The derivative of the sigmoid of 0 is %f\n", sigmoid(0,true));
	feedForward(p_input, p_weights, p_output, p_layerSizes, layerCount);
	return 0;
}
