/**The MIT License (MIT)
*
*Copyright (c) 2016 Dylan
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/

#include "../include/simulation_data.hpp"

#include <stdlib.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>

#include "mkl.h"

#include "../include/save_data.hpp"

SimulationData::SimulationData(int num_points) {
	this->num_points = num_points;
	this->x_length = 200;
	this->x = (double*)mkl_malloc(this->num_points * sizeof(double), 64);
	this->num_real_steps = 100000;
	this->beta = 1;
	this->a1 = 0;	

	double xval = 0;
	#pragma omp parallel for private(xval)
	for (int i = 0; i < this->num_points; ++i) {
		xval = -0.5 * this->x_length + i * this->x_length / (this->num_points - 1);
		this-> x[i] = xval;
	}
	this-> dx = this->x[1] - this->x[0];
	this->dx2 = pow(this->dx, 2.0);
	this->dt = 0.0001 * this->dx;
}

SimulationData::~SimulationData() {
	mkl_free(x);
}
