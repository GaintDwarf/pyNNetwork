#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "NNetwork.h"


// ***************************** extra functions ********************************

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
	double y = sigmoid(x);
	return y * (1.0 - y);
}

double quadraticCost(Matrix* result, Matrix* output) {
	double cost = 0, temp;
	for (int i = 0; i < result->getCols(); i++)
	{
		temp = result->get(i, 0) - output->get(i, 0);
		cost += temp * temp;
	}
	return cost * 0.5;
}

Matrix* quadraticDerivative(Matrix* result, Matrix* output) {
	return result->sub(output);
}

int maxIndex(Array searchIn) {
	int imax = 0;
	for (int i = 1; i < searchIn.length; i++)
	{
		if (searchIn.arr[i] > searchIn.arr[imax]) {
			imax = i;
		}
	}
	return imax;
}

int maxIndex(py::list searchIn) {
	int imax = 0;
	for (int i = 1; i < searchIn.size(); i++)
	{
		if (searchIn[i].cast<double>() > searchIn[imax].cast<double>()) {
			imax = i;
		}
	}
	return imax;
}

int maxIndex(std::vector<double> &searchIn) {
	int imax = 0;
	for (int i = 1; i < searchIn.size(); i++)
	{
		if (searchIn[i] > searchIn[imax]) {
			imax = i;
		}
	}
	return imax;
}

void shuffel(Array *set1, Array *set2, int length) {
	int max = length, newInd;
	srand(time(NULL));

	for (int i = 0; i < length; i++)
	{
		newInd = rand() % max;
		Array temp1 = set1[i];
		set1[i] = set1[newInd];
		set1[newInd] = temp1;
		Array temp2 = set2[i];
		set2[i] = set2[newInd];
		set2[newInd] = temp2;
	}
}

void shuffle(py::list set1, py::list set2) {
	int max = set1.size(), newInd;
	srand(time(NULL));

	for (int i = 0; i < set1.size(); i++)
	{
		newInd = rand() % max;
		py::list temp1 = set1[i].cast<py::list>();
		set1[i] = set1[newInd].cast<py::list>();
		set1[newInd] = temp1;
		py::list temp2 = set2[i].cast<py::list>();
		set2[i] = set2[newInd].cast<py::list>();
		set2[newInd] = temp2;
	}
}

// ************************* neural network functions ****************************

NNetwork::NNetwork()
{
}

NNetwork::NNetwork(int* layers, int size) {

	this->size = size;
	this->layers = new int[size];
	for (int i = 0; i < size; i++)
	{
		this->layers[i] = layers[i];
	}

	this->wights = new Matrix*[(size - 1)];
	this->biases = new Matrix*[(size - 1)];

	// initiate the matrices
	for (int i = 0; i < size - 1; i++)
	{
		this->wights[i] = new Matrix(layers[i + 1], layers[i]);
		this->biases[i] = new Matrix(layers[i + 1], 1);
	}

	this->activation = sigmoid;
	this->activationPrime = sigmoidDerivative;

	this->cost = quadraticCost;
	this->costDerivative = quadraticDerivative;

	this->RandomizeMatrices();
}

NNetwork::NNetwork(py::list layers)
{
	this->size = layers.size();
	this->layers = new int[size];
	for (int i = 0; i < size; i++)
	{
		this->layers[i] = layers[i].cast<int>();
	}

	this->wights = new Matrix*[(size - 1)];
	this->biases = new Matrix*[(size - 1)];

	// initiate the matrices
	for (int i = 0; i < size - 1; i++)
	{
		this->wights[i] = new Matrix(this->layers[i + 1], this->layers[i]);
		this->biases[i] = new Matrix(this->layers[i + 1], 1);
	}

	this->activation = sigmoid;
	this->activationPrime = sigmoidDerivative;

	this->cost = quadraticCost;
	this->costDerivative = quadraticDerivative;

	this->RandomizeMatrices();
}

void NNetwork::RandomizeMatrices() {
	srand(time(NULL));

	for (int i = 0; i < size - 1; i++)
	{
		this->wights[i]->randomizePreTimeInit(-5, 5);
		this->biases[i]->randomizePreTimeInit(-5, 5);
	}

}

void NNetwork::printLayers() {

	for (int i = 0; i < size - 1; i++)
	{
		std::cout << std::endl << "wights " << i + 1 << " ==> " << i + 2 << std::endl;
		this->wights[i]->printMat();
	}

}

void NNetwork::freelayers() {

	// initiate the matrices
	for (int i = 0; i < size - 1; i++)
	{
		delete this->wights[i];
		delete this->biases[i];
	}
	delete[] this->wights;
	delete[] this->biases;

	delete[] this->layers;
}

NNetwork::~NNetwork()
{
	this->freelayers();
}

// ********************************* AI functions ********************************

/// ------------------------------------------------------------------------------
/// feedforword
/// ------------------------------------------------------------------------------
/// <summary>
///		Feedforwords the specified input 
///		using the feedforword algorithem of neural network.
/// </summary>
/// <param name="input">	The input array.				</param>
/// <param name="length">	The length of the input array.	</param>
/// <returns>
///		the output for the given input
///</returns>
/// ------------------------------------------------------------------------------
std::vector<double> NNetwork::feedforword(Array arr) {

	if (layers[0] != arr.length) {
		throw "the input layer length does not match the input";
	}

	Matrix *last = NULL, *result = NULL;

	// initiate the first layer
	result = new Matrix(arr.arr, arr.length);

	last = this->wights[0]->multiply(result);
	delete result;

	result = last;
	result->addSelf(this->biases[0]);

	result->mapSelf(this->activation);

	// the rest of the layers
	for (int i = 1; i < this->size - 1; i++)
	{
		last = this->wights[i]->multiply(result);
		delete result;
		result = last;
		result->addSelf(this->biases[i]);

		result->mapSelf(activation);
	}

	std::vector<double> temparr = result->toVector();
	delete result;

	return temparr;
}

std::vector<double> NNetwork::feedforword(py::list arr)
{
	if (layers[0] != arr.size()) {
		throw "the input layer length does not match the input";
	}

	Matrix *last = NULL, *result = NULL;

	// initiate the first layer
	result = new Matrix(arr);
	last = this->wights[0]->multiply(result);
	delete result;

	result = last;
	result->addSelf(this->biases[0]);
	result->mapSelf(this->activation);

	// the rest of the layers
	for (int i = 1; i < this->size - 1; i++)
	{
		last = this->wights[i]->multiply(result);
		delete result;
		result = last;
		result->addSelf(this->biases[i]);
		result->mapSelf(activation);
	}

	std::vector<double> temparr = result->toVector();
	delete result;

	return temparr;
}


/// ------------------------------------------------------------------------------
/// SGradientDescent
/// ------------------------------------------------------------------------------
/// <summary>
///		The function trains the neural network using stochastic gradiant desend.
/// </summary>
/// <param name="inputs">		The input arrays to train by.			</param>
/// <param name="outputs">		The solutions to the input array 
///								(need to match the order of the inputs).</param>
/// <param name="batchSize">	Size of the batch.						</param>
/// <param name="learningRate">	The learning rate.						</param>
/// <param name="epochs">		How many time to repeat the process		</param>
/// <param name="testInputs">	The input arrays to test by.			</param>
/// <param name="testOutputs">	The solutions to the test input array 
///								(need to match the order of the inputs).</param>
/// <param name="testSize">		Size of the test.						</param>
/// ------------------------------------------------------------------------------
void NNetwork::SGradientDescent(Array *inputs, Array *outputs, int length, int batchSize, 
	double learningRate, int epochs, Array* testInputs, Array* testOutputs, int testSize){

	for (int i = 0; i < epochs; i++)
	{
		shuffel(inputs, outputs, length);
		for (int setStart = 0; setStart < length - batchSize + 1; setStart += batchSize)
		{
			updateBatch(&inputs[setStart], &outputs[setStart], batchSize, learningRate);
		}
		int passed = evaluate(testInputs, testOutputs, testSize);
		printf("Gen %d : %d / %d passed\n", i, passed, testSize);
	}
}

void NNetwork::SGradientDescent(py::list inputs, py::list outputs, int batchSize, double learningRate,
	int epochs, py::list testInputs, py::list testOutputs) {

	for (int i = 0; i < epochs; i++)
	{
		shuffle(inputs, outputs);
		for (int setStart = 0; setStart < inputs.size() - batchSize + 1; setStart += batchSize)
		{
			updateBatch(&inputs, &outputs, setStart, batchSize, learningRate);
		}
		int passed = evaluate(testInputs, testOutputs);
		printf("Gen %d : %d / %d passed\n", i, passed, testInputs.size());
	}

}


/// ------------------------------------------------------------------------------
/// SGradientDescent
/// ------------------------------------------------------------------------------
/// <summary>
///		The function trains the neural network using stochastic gradiant desend.
/// </summary>
/// <param name="inputs">		The input arrays to train by.			</param>
/// <param name="outputs">		The solutions to the input array 
///								(need to match the order of the inputs).</param>
/// <param name="batchSize">	Size of the batch.						</param>
/// <param name="learningRate">	The learning rate.						</param>
/// <param name="epochs">		How many time to repeat the process		</param>
/// ------------------------------------------------------------------------------
void NNetwork::SGradientDescent(py::list inputs, py::list outputs, int batchSize,
	double learningRate, int epochs) {

	for (int i = 0; i < epochs; i++)
	{
		shuffle(inputs, outputs);
		for (int setStart = 0; setStart < inputs.size() - batchSize + 1; setStart += batchSize)
		{
			updateBatch(&inputs, &outputs, setStart, batchSize, learningRate);
		}
	}
}

void NNetwork::SGradientDescent(Array *inputs, Array *outputs, int length, int batchSize,
	double learningRate, int epochs) {

	for (int i = 0; i < epochs; i++)
	{
		shuffel(inputs, outputs, length);
		for (int setStart = 0; setStart < length - batchSize + 1; setStart += batchSize)
		{
			updateBatch(&inputs[setStart], &outputs[setStart], batchSize, learningRate);
		}
	}
}


/// ------------------------------------------------------------------------------
/// backpropagate
/// ------------------------------------------------------------------------------
/// <summary>
///		The function implements the backpropagtion, it feed the network and then
///		calculating the error for every layer.
///		the deltas need to arive freed and after being used be to freed again. 
/// </summary>
/// <param name="input">			The input to feed in.					</param>
/// <param name="output">			The desired output.						</param>
/// <param name="deltaWights">		The place to store delta of the wights.	</param>
/// <param name="deltaBiases">		The place to store delta of the biases.	</param>
/// <param name="length">			The length of the deltas.				</param>
/// ------------------------------------------------------------------------------
void NNetwork::backpropagate(Array input, Array output, 
	Matrix **deltaWights[], Matrix **deltaBiases[], int *length)
{
	if (input.length != this->layers[0] || output.length != this->layers[size - 1]) {
		throw "the length of the input or output doesn't match the neural network";
	}
	// alloact the deltas
	*deltaWights = new Matrix*[size - 1];
	*deltaBiases = new Matrix*[size - 1];
	*length = (size - 1);

	// feedforword
	Matrix *last = NULL, *result = NULL, *mid = NULL, *delta = NULL, *trans = NULL;
	Matrix **non_activated_layers = new Matrix*[this->size];
	Matrix **activated_layers = new Matrix*[this->size];

	// initiate the first layer
	last = new Matrix(input.arr, input.length);
	non_activated_layers[0] = last;
	result = last->map(this->activation);
	activated_layers[0] = result;

	// the rest of the layers
	for (int i = 0; i < this->size - 1; i++)
	{
		last = this->wights[i]->multiply(result);
		last->addSelf(this->biases[i]);

		non_activated_layers[i + 1] = last;
		result = last->map(this->activation);
		activated_layers[i + 1] = result;
	}

	// start of the backpropagation
	mid = new Matrix(output.arr, output.length);			// created new matrix (not importent)
	delta = costDerivative(result, mid);					// created new matrix (importent)
	delete mid;
	mid = non_activated_layers[size - 1]->map(this->activationPrime); // changed to non and added map

	delta->hadamardSelf(mid);
	delete mid;
	(*deltaBiases)[size - 2] = delta;
	trans = activated_layers[size - 2]->transpose();		// created new matrix (not importent)
	(*deltaWights)[size - 2] = delta->multiply(trans);		// create new matrix (importent)
	delete trans;

	// beginning of the backpropagtion loop
	for (int layer = size - 2; layer > 0; layer--)
	{
		last = non_activated_layers[layer];
		mid = last->map(activationPrime);					// created new matrix (not importent)
		trans = wights[layer]->transpose();					// created new matrix (not importent)
		last = trans->multiply(delta);						// created new matrix (not importent)
		delete trans;
		delta = last->hadamard(mid);						// create new matrix (imprtent)
		delete last;
		delete mid;
		(*deltaBiases)[layer - 1] = delta;
		trans = activated_layers[layer - 1]->transpose();	// created new matrix (not importent)
		(*deltaWights)[layer - 1] = delta->multiply(trans);
		delete trans;
	}

	// release the crearted array
	for (int i = 0; i < this->size; i++)
	{
		delete non_activated_layers[i];
		delete activated_layers[i];
	}
	delete[] non_activated_layers;
	delete[] activated_layers;
}

void NNetwork::backpropagate(py::list input, py::list output, Matrix ** deltaWights[], 
	Matrix ** deltaBiases[], int * length)
{
	if (input.size() != this->layers[0] || output.size() != this->layers[size - 1]) {
		throw "the length of the input or output doesn't match the neural network";
	}
	// alloact the deltas
	*deltaWights = new Matrix*[size - 1];
	*deltaBiases = new Matrix*[size - 1];
	*length = (size - 1);

	// feedforword
	Matrix *last = NULL, *result = NULL, *mid = NULL, *delta = NULL, *trans = NULL;
	Matrix **non_activated_layers = new Matrix*[this->size];
	Matrix **activated_layers = new Matrix*[this->size];

	// initiate the first layer
	last = new Matrix(input);
	non_activated_layers[0] = last;
	result = last->map(this->activation);
	activated_layers[0] = result;

	// the rest of the layers
	for (int i = 0; i < this->size - 1; i++)
	{
		last = this->wights[i]->multiply(result);
		last->addSelf(this->biases[i]);

		non_activated_layers[i + 1] = last;
		result = last->map(this->activation);
		activated_layers[i + 1] = result;
	}

	// start of the backpropagation
	mid = new Matrix(output);								// created new matrix (not importent)
	delta = costDerivative(result, mid);					// created new matrix (importent)
	delete mid;
	mid = non_activated_layers[size - 1]->map(this->activationPrime); // changed to non and added map

	delta->hadamardSelf(mid);
	delete mid;
	(*deltaBiases)[size - 2] = delta;
	trans = activated_layers[size - 2]->transpose();		// created new matrix (not importent)
	(*deltaWights)[size - 2] = delta->multiply(trans);		// create new matrix (importent)
	delete trans;

	// beginning of the backpropagtion loop
	for (int layer = size - 2; layer > 0; layer--)
	{
		last = non_activated_layers[layer];
		mid = last->map(activationPrime);					// created new matrix (not importent)
		trans = wights[layer]->transpose();					// created new matrix (not importent)
		last = trans->multiply(delta);						// created new matrix (not importent)
		delete trans;
		delta = last->hadamard(mid);						// create new matrix (imprtent)
		delete last;
		delete mid;
		(*deltaBiases)[layer - 1] = delta;
		trans = activated_layers[layer - 1]->transpose();	// created new matrix (not importent)
		(*deltaWights)[layer - 1] = delta->multiply(trans);
		delete trans;
	}

	// release the crearted array
	for (int i = 0; i < this->size; i++)
	{
		delete non_activated_layers[i];
		delete activated_layers[i];
	}
	delete[] non_activated_layers;
	delete[] activated_layers;
}



/// ------------------------------------------------------------------------------
/// update batch
/// ------------------------------------------------------------------------------
/// <summary>
///		Updates the batch.
/// </summary>
/// <param name="inputs">			The input arrays to train by.				</param>
/// <param name="outputs">			The solutions to the input array 
///									(need to match the order of the inputs).	</param>
/// <param name="batchSize">		Size of the batche.							</param>
/// <param name="learningRate">		The learning rate.							</param>
/// ------------------------------------------------------------------------------
void NNetwork::updateBatch(Array *inputs, Array *outputs, int batchSize, double learningRate) {

	if (batchSize > 0) {
		int changedLength;

		Matrix **deltaWights;
		Matrix **deltaBiases;
		this->backpropagate(inputs[0], outputs[0], &deltaWights, &deltaBiases, &changedLength);

		Matrix **tempWDelta;
		Matrix **tempBDelta;

		// figure out the deltas
		for (int i = 1; i < batchSize; i++)
		{
			this->backpropagate(inputs[i], outputs[i], &tempWDelta, &tempBDelta, &changedLength);
			for (int j = 0; j < changedLength; j++)
			{
				deltaWights[j]->addSelf(tempWDelta[j]);
				deltaBiases[j]->addSelf(tempBDelta[j]);
				delete tempWDelta[j];
				delete tempBDelta[j];
			}
			delete[] tempWDelta;
			delete[] tempBDelta;
		}

		// update the wights and biases
		for (int i = 0; i < this->size - 1; i++)
		{
			deltaWights[i]->multiplySelf(learningRate / 100.0);
			deltaBiases[i]->multiplySelf(learningRate / 100.0);

			wights[i]->subSelf(deltaWights[i]);
			biases[i]->subSelf(deltaBiases[i]);
			// cleanup
			delete deltaWights[i];
			delete deltaBiases[i];
		}

		// cleanup
		delete[] deltaBiases;
		delete[] deltaWights;
	}
}

void NNetwork::updateBatch(py::list *inputs, py::list *outputs, int startAt, int size, double learningRate)
{
	if (size > 0) {
		int changedLength;

		Matrix **deltaWights;
		Matrix **deltaBiases;
		this->backpropagate((*inputs)[startAt], (*outputs)[startAt], &deltaWights, &deltaBiases, &changedLength);

		Matrix **tempWDelta;
		Matrix **tempBDelta;

		// figure out the deltas
		for (int i = startAt + 1; i < startAt + size - 1 && i < (*inputs).size(); i++)
		{
			this->backpropagate((*inputs)[i], (*outputs)[i], &tempWDelta, &tempBDelta, &changedLength);
			for (int j = 0; j < changedLength; j++)
			{
				deltaWights[j]->addSelf(tempWDelta[j]);
				deltaBiases[j]->addSelf(tempBDelta[j]);
				delete tempWDelta[j];
				delete tempBDelta[j];
			}
			delete[] tempWDelta;
			delete[] tempBDelta;
		}

		// update the wights and biases
		for (int i = 0; i < this->size - 1; i++)
		{
			deltaWights[i]->multiplySelf(learningRate / 100.0);
			deltaBiases[i]->multiplySelf(learningRate / 100.0);

			wights[i]->subSelf(deltaWights[i]);
			biases[i]->subSelf(deltaBiases[i]);
			// cleanup
			delete deltaWights[i];
			delete deltaBiases[i];
		}

		// cleanup
		delete[] deltaBiases;
		delete[] deltaWights;
	}
}


/// ------------------------------------------------------------------------------
/// evaluate
/// ------------------------------------------------------------------------------
/// <summary>
///		Evaluates the specified test data.
/// </summary>
/// <param name="inputs">	The input arrays to test by.				</param>
/// <param name="outputs">	The solutions to the input array 
///							(need to match the order of the inputs).	</param>
/// <param name="testSize">	Size of the test.							</param>
/// <returns> 
///		how many test passed	
/// </returns>
/// ------------------------------------------------------------------------------
int NNetwork::evaluate(Array* inputs, Array* outputs, int testSize)
{
	int counter = 0;
	std::vector<double> res;

	for (int i = 0; i < testSize; i++)
	{
		res = this->feedforword(inputs[i]);
		counter += (maxIndex(res) == maxIndex(outputs[i]));
	}

	return counter;
}

int NNetwork::evaluate(py::list inputs, py::list outputs)
{
	int counter = 0;
	std::vector<double> res;

	for (int i = 0; i < inputs.size(); i++)
	{
		res = this->feedforword(inputs[i].cast<py::list>());
		counter += (maxIndex(res) == maxIndex(outputs[i]));
	}

	return counter;
}


//**************************************** Files **************************************
NNetwork::NNetwork(char *fileName) {
	// openning the file
	FILE* fl = fopen(fileName, "r");
	fscanf(fl, "size: %d\n", &size);

	layers = new int[size];

	for (int i = 0; i < size; i++)
	{
		fscanf(fl, "%d, ", &layers[i]);
	}
	fscanf(fl, "\n");

	this->wights = new Matrix*[(size - 1)];
	this->biases = new Matrix*[(size - 1)];

	for (int i = 0; i < size - 1; i++)
	{
		fscanf(fl, "\nwights at %d:\n", &i);
		this->wights[i] = new Matrix(layers[i + 1], layers[i]);
		wights[i]->readMatFromFile(fl);
		fscanf(fl, "\nbiases at %d:\n", &i);
		this->biases[i] = new Matrix(layers[i + 1], 1);
		biases[i]->readMatFromFile(fl);
	}

	fclose(fl);
}

NNetwork::NNetwork(const char *fileName) {
	// openning the file
	FILE* fl = fopen(fileName, "r");
	fscanf(fl, "size: %d\n", &size);

	layers = new int[size];

	for (int i = 0; i < size; i++)
	{
		fscanf(fl, "%d, ", &layers[i]);
	}
	fscanf(fl, "\n");

	this->wights = new Matrix*[(size - 1)];
	this->biases = new Matrix*[(size - 1)];

	for (int i = 0; i < size - 1; i++)
	{
		fscanf(fl, "\nwights at %d:\n", &i);
		this->wights[i] = new Matrix(layers[i + 1], layers[i]);
		wights[i]->readMatFromFile(fl);
		fscanf(fl, "\nbiases at %d:\n", &i);
		this->biases[i] = new Matrix(layers[i + 1], 1);
		biases[i]->readMatFromFile(fl);
	}

	fclose(fl);

	this->activation = sigmoid;
	this->activationPrime = sigmoidDerivative;

	this->cost = quadraticCost;
	this->costDerivative = quadraticDerivative;
}

void NNetwork::saveNetworkTxt(const char *fileName) {
	// openning the file
	FILE* fl = fopen(fileName, "w");
	fprintf(fl, "size: %d\n", size);
	for (int i = 0; i < size; i++)
	{
		fprintf(fl, "%d, ", layers[i]);
	}
	fprintf(fl, "\n");

	for (int i = 0; i < size - 1; i++)
	{
		fprintf(fl, "\nwights at %d:\n", i);
		wights[i]->printMatToFile(fl);
		fprintf(fl, "\nbiases at %d:\n", i);
		biases[i]->printMatToFile(fl);
	}

	fclose(fl);

	this->activation = sigmoid;
	this->activationPrime = sigmoidDerivative;

	this->cost = quadraticCost;
	this->costDerivative = quadraticDerivative;
}
