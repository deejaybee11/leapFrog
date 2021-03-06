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

#include "../include/potential_data.hpp"

#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "mkl.h"

#include "../include/simulation_data.hpp"
#include "../include/wavefunction_data.hpp"
#include "../include/save_data.hpp"

PotentialData::PotentialData(SimulationData &sim_data) {
	this->harmonic_trap = (double*)mkl_malloc(sim_data.num_points * sizeof(double), 64);

	#pragma omp parallel for
	for (int i = 0; i < sim_data.num_points; ++i) {
		harmonic_trap[i] = 0.5 * pow(sim_data.x[i], 2.0);
	}
}

PotentialData::~PotentialData() {
	mkl_free(harmonic_trap);
}
