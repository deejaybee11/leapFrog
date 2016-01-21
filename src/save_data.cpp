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

#include "../include/save_data.hpp"

#include <stdio.h>

#include "mkl.h"

#include "../include/simulation_data.hpp"


int save_data(double *data, SimulationData &sim_data, const char * filename) {
	FILE* pFile;
	pFile = fopen(filename, "wb");
	fwrite(data, sizeof(double), sim_data.num_points, pFile);
	fclose(pFile);
	return 0;
}

int save_data_real(MKL_Complex16 *data, SimulationData &sim_data, const char * filename) {
	double *data2;
	data2 = (double*)mkl_malloc(sim_data.num_points * sizeof(double), 64);

	for (int i = 0; i < sim_data.num_points; ++i) {
		data2[i] = data[i].real;
	}

	FILE* pFile;
	pFile = fopen(filename, "wb");
	fwrite(data2, sizeof(double), sim_data.num_points, pFile);
	fclose(pFile);
	return 0;
}
