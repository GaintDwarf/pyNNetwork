#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include "Matrix.h"

namespace py = pybind11;

Matrix::Matrix()
{
}

Matrix::Matrix(double *arr, int length) {
	rows = length;
	cols = 1;
	table = new double*[rows];

	for (int i = 0; i < rows; i++)
	{
		table[i] = (double*)malloc(sizeof(double) * cols);
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = arr[i];
		}
	}
};

Matrix::Matrix(py::list arr) {
	rows = arr.size();
	cols = 1;
	table = new double*[rows];

	for (int i = 0; i < rows; i++)
	{
		table[i] = (double*)malloc(sizeof(double) * cols);
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = arr[i].cast<double>();
		}
	}
};

Matrix::Matrix(int rowCount, int colCount) {
	rows = rowCount;
	cols = colCount;
	table = new double*[rowCount];

	for (int i = 0; i < rowCount; i++)
	{
		table[i] = new double[colCount];
		for (int j = 0; j < colCount; j++)
		{
			table[i][j] = 0;
		}
	}
}

Matrix::Matrix(int rowCount, int colCount, double* arr)
{
	rows = rowCount;
	cols = colCount;
	table = new double*[rowCount];

	for (int i = 0; i < rowCount; i++)
	{
		table[i] = new double[colCount];
		for (int j = 0; j < colCount; j++)
		{
			table[i][j] = arr[cols * i + j];
		}
	}
}

Matrix::Matrix(int rowCount, int colCount, int* arr)
{
	rows = rowCount;
	cols = colCount;
	table = new double*[rowCount];

	for (int i = 0; i < rowCount; i++)
	{
		table[i] = new double[colCount];
		for (int j = 0; j < colCount; j++)
		{
			table[i][j] = arr[cols * i + j];
		}
	}
}

Matrix::Matrix(int rowCount, int colCount, py::list arr)
{
	rows = rowCount;
	cols = colCount;
	table = new double*[rowCount];

	for (int i = 0; i < rowCount; i++)
	{
		table[i] = new double[colCount];
		for (int j = 0; j < colCount; j++)
		{
			table[i][j] = arr[cols * i + j].cast<double>();
		}
	}
}

Matrix::Matrix(int rowCount, int colCount, double initValue) {
	rows = rowCount;
	cols = colCount;
	table = new double*[rowCount];

	for (int i = 0; i < rowCount; i++)
	{
		table[i] = new double[colCount];
		for (int j = 0; j < colCount; j++)
		{
			table[i][j] = initValue;
		}
	}
};

Matrix::~Matrix()
{
	for (int i = 0; i < rows; i++)
	{
		delete[] table[i];
	}
	delete[] table;
};

void Matrix::randomize(double min, double max) 
{
	srand(time(NULL));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = min + ((double)rand() / RAND_MAX) * (max - min);
		}
	}
}

void Matrix::randomizePreTimeInit(double min, double max)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = min + ((double)rand() / RAND_MAX) * (max - min);
		}
	}
}

void Matrix::printMat()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%-10.6f", table[i][j]);
		}
		std::cout << ' ' << std::endl;
	}
}

void Matrix::printMatToFile(FILE *fl) {

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(fl, "%-50.45f ", table[i][j]);
		}
		fprintf(fl, " \n");
	}
}

void Matrix::readMatFromFile(FILE *fl) {

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(fl, "%lf ", &table[i][j]);
		}
		fscanf(fl, " \n");
	}
}

void Matrix::freeTable()
{
	for (int i = 0; i < rows; i++)
	{
		delete[] table[i];
	}
	delete[] table;
}

Matrix* Matrix::transpose()
{
	Matrix *newm = new Matrix(cols, rows);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[j][i] = table[i][j];
		}
	}
	return newm;
}

Matrix* Matrix::map(double activation(double x)) {

	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = activation(table[i][j]);
		}
	}
	return newm;

}

void Matrix::mapSelf(double activation(double x)) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = activation(table[i][j]);
		}
	}
}

Matrix* Matrix::add(double scalar)
{
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] + scalar;
		}
	}
	return newm;
}

Matrix* Matrix::add(Matrix *other)
{
	if (other->rows != rows || other->cols != cols) {
		throw "The addition is elementwise you must send matrices which have the same dimentions";
	}
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] + other->table[i][j];
		}
	}
	return newm;
}

void Matrix::addSelf(double scalar){
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] + scalar;
		}
	}
}

void Matrix::addSelf(Matrix *other) {

	if (other->rows != rows || other->cols != cols) {
		throw "The addition is elementwise you must send matrices which have the same dimentions";
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] + other->table[i][j];
		}
	}

}

Matrix* Matrix::sub(double scalar)
{
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] - scalar;
		}
	}
	return newm;
}

Matrix* Matrix::sub(Matrix *other)
{
	if (other->rows != rows || other->cols != cols) {
		throw "The subtraction is elementwise you must send matrices which have the same dimentions";
	}
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] - other->table[i][j];
		}
	}
	return newm;
}

void Matrix::subSelf(double scalar) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] - scalar;
		}
	}
}

void Matrix::subSelf(Matrix *other) {

	if (other->rows != rows || other->cols != cols) {
		throw "The subtraction is elementwise you must send matrices which have the same dimentions";
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] - other->table[i][j];
		}
	}

}

Matrix* Matrix::multiply(double scalar)
{
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] * scalar;
		}
	}
	return newm;
}

Matrix* Matrix::multiply(Matrix *other)
{
	if (other->rows != cols) {
		throw "In order to multiply matrices the cols of the matrix must have the same rows as the sent matrix";
	}
	Matrix *newm = new Matrix(rows, other->cols);
	double sum;
	for (int col = 0; col < other->cols; col++)
	{
		for (int row = 0; row < rows; row++)
		{
			sum = 0;
			for (int index = 0; index < cols; index++)
			{
				sum += table[row][index] * other->table[index][col];
			}
			newm->table[row][col] = sum;
		}
	}
	return newm;
}

Matrix* Matrix::hadamard(Matrix *other)
{
	if (other->rows != rows || other->cols != cols) {
		throw "The hadamard multipliction is elementwise you must send matrices which have the same dimentions";
	}
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] * other->table[i][j];
		}
	}
	return newm;
}

void Matrix::multiplySelf(double scalar) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] * scalar;
		}
	}
}

void Matrix::hadamardSelf(Matrix *other) {

	if (other->rows != rows || other->cols != cols) {
		throw "The hadamrd function is elementwise you must send matrices which have the same dimentions";
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] * other->table[i][j];
		}
	}

}

Matrix* Matrix::div(double scalar)
{
	Matrix *newm = new Matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newm->table[i][j] = table[i][j] / scalar;
		}
	}
	return newm;
}

void Matrix::divSelf(double scalar) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			table[i][j] = table[i][j] / scalar;
		}
	}
}

double* Matrix::toArray()
{
	double* newarr = new double[rows * cols];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newarr[i * cols + cols] = table[i][j];
		}
	}
	return newarr;
}

std::vector<double> Matrix::toVector() {

	std::vector<double> newarr;
	newarr.reserve(rows * cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			newarr.push_back(table[i][j]);
		}
	}
	return newarr;

}
