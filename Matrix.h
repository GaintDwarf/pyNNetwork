#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <vector>

namespace py = pybind11;

class Matrix
{
protected:
	int rows;
	int cols;
	double **table;
public:
	Matrix();
	Matrix(double *arr, int length);
	Matrix(py::list arr);
	Matrix(int rowCount, int colCount);
	Matrix(int rowCount, int colCount, double *arr);
	Matrix(int rowCount, int colCount, int * arr);
	Matrix(int rowCount, int colCount, py::list arr);
	Matrix(int rowCount, int colCount, double initValue);
	~Matrix();

	double **getTable()			{ return table; };
	int getRows()				{ return rows; };
	int getCols()				{ return cols; };
	double get(int x, int y)	{ 
		if(x < rows && x >= 0 && y < cols && y >= 0) return table[x][y];
		else throw "out of range";
	};
	// general
	void randomize(double min, double max);
	void randomizePreTimeInit(double min, double max);
	void printMat();
	void printMatToFile(FILE *fl);
	void readMatFromFile(FILE *fl);
	void freeTable();
	Matrix* transpose();
	Matrix* map(double activation(double x));
	void mapSelf(double activation(double x));
	// addition
	Matrix* add(double scalar);
	Matrix* add(Matrix *other);
	void addSelf(double scalar);
	void addSelf(Matrix *other);
	// subttraction
	Matrix* sub(double scalar);
	Matrix* sub(Matrix *other);
	void subSelf(double scalar);
	void subSelf(Matrix *other);
	// multiplication
	Matrix* multiply(double scalar);
	Matrix* multiply(Matrix *other);
	Matrix* hadamard(Matrix *other);
	void multiplySelf(double scalar);
	void hadamardSelf(Matrix *other);
	// divition
	Matrix* div(double scalar);
	void divSelf(double scalar);
	// Matrix -> array
	double* toArray();
	std::vector<double> toVector();
};

