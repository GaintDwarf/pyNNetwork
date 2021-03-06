#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <math.h>
#include "Matrix.h"
#include "NNetwork.h"

namespace py = pybind11;

//  
//                                    /   \       
//   _                        )      ((   ))     (
//  (@)                      /|\      ))_((     /|\
//  |-|                     / | \    (/\|/\)   / | \                      (@)
//  | | -------------------/--|-voV---\`|'/--Vov-|--\---------------------|-|
//  |-|                         '^`   (o o)  '^`                          | |
//  | |                               `\Y/'                               |-|
//  |-|                                                                   | |
//  | |                Project Name: M.NI                                 | |
//  | |                Author: Segev Gershon                              |-|
//  | |                Date of Complition: N/A                            | |
//  | |                Version: 0.7                                       | |
//  |-|                                                                   | |
//  | |    Description : Neural Network library for python writen in C++  | |
//  | |    Classes     : -NNetwork                                        | |
//  | |    Functions   :                                                  | |
//  |-|     NNetwork :                                                    | |
//  | |     - init(layers list / file name)                               | |
//  | |           the funcion initiates the network with the sent layers  | |
//  | |           when each layer is a index and containts how many       | |
//  | |           neurons int the layer or the file name of the saved     |-|
//  | |           network.                                                | |
//  | |     - print layers ()                                             | |
//  |-|           the function prints the layers.                         |-|
//  | |     - feedforword (input)                                         | |
//  | |           the function feeds forword the input in the network.    | |
//  | |           the input is in the form of list when each cell is the  |-|
//  | |           value of the starting neuron                            | |
//  | |     - save_txt (name)                                             | |
//  |-|            the function save the neuarl network to a .txt file.   |-|
//  | |            send with postfix.                                     | |
//  | |     - SGradient_descent(inputs, outputs, batch_size,              | |
//  | |                 learning_rate, epochs, test_inputs, test_outputs) | |
//  |-|             the function teaches the network to the input         | |
//  | |                                                                   |-|
//  |_|___________________________________________________________________| |
//  (@)              l   /\ /         ( (       \ /\   l                `\|-|
//                   l /   V           \ \       V   \ l                  (@)
//                   l/                _) )_          \I
//                                     `\ /'
//  


PYBIND11_MODULE(pyNNetwork, m) {

	py::class_<NNetwork>(m, "NNetwork")
		.def(py::init<py::list>())
		.def(py::init<const char*>())
		.def(py::init<char *>())
		.def("print_layers", &NNetwork::printLayers)
		.def("feedforword", py::overload_cast<py::list>(&NNetwork::feedforword))
		.def("SGradient_descent", py::overload_cast<py::list, py::list, int, double, int, py::list, py::list>
			(&NNetwork::SGradientDescent))
		.def("SGradient_descent", py::overload_cast<py::list, py::list, int, double, int>(&NNetwork::SGradientDescent))
		.def("save_txt", &NNetwork::saveNetworkTxt);
}

