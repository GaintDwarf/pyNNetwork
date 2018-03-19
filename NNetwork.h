#pragma once
#include <vector>
#include <pybind11/stl.h>
#include "Matrix.h"

typedef struct Array {
	double* arr;
	int length;
}Array;

class NNetwork
{
protected:
	// parameters
	int size;
	int* layers;
	Matrix** wights;
	Matrix** biases;
	// functions
	double (*activation)		(double num);
	double (*activationPrime)	(double num);
	double (*cost)				(Matrix* result, Matrix* output);
	Matrix* (*costDerivative)	(Matrix* result, Matrix* output);

public:
	// general function
	NNetwork();
	NNetwork(int* layers, int size);
	NNetwork(py::list layers);
	void RandomizeMatrices();
	void printLayers();
	void freelayers();
	~NNetwork();
	// ai functions
	std::vector<double> feedforword(Array arr);
	std::vector<double> feedforword(py::list arr);
	void SGradientDescent(Array *inputs, Array *outputs, int length, int batchSize, double learningRate,
		int epochs, Array* testInputs, Array* testOutputs, int testSize);
	void SGradientDescent(Array *inputs, Array *outputs, int length, int batchSize, double learningRate, 
		int epochs);
	void SGradientDescent(py::list inputs, py::list outputs, int batchSize, double learningRate,
		int epochs, py::list testInputs, py::list testOutputs);
	void SGradientDescent(py::list inputs, py::list outputs, int batchSize, double learningRate,
		int epochs);
	void updateBatch(Array *inputs, Array *outputs, int batchSize, double learningRate);
	void updateBatch(py::list *inputs, py::list *outputs, int startAt, int size, double learningRate);
	void backpropagate(Array input, Array output,
		Matrix **deltaWights[], Matrix **deltaBiases[], int *length);
	void backpropagate(py::list input, py::list output,
		Matrix **deltaWights[], Matrix **deltaBiases[], int *length);
	int evaluate(Array *inputs, Array *outputs, int testSize);
	int evaluate(py::list inputs, py::list outputs);
	// file functions
	NNetwork(char *fileName);
	NNetwork(const char *fileName);
	void saveNetworkTxt(const char *fileName);
	void saveNetworkBin(const char *fileName);
};

