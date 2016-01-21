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

#include "../include/wavefunction_data.hpp"

#include <stdlib.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>

#include "mkl.h"

#include "../include/simulation_data.hpp"
#include "../include/save_data.hpp"

WavefunctionData::WavefunctionData(SimulationData &sim_data) {
	this->psi = (MKL_Complex16*)mkl_malloc(sim_data.num_points * sizeof(MKL_Complex16), 64);
	this->psi_old = (MKL_Complex16*)mkl_malloc(sim_data.num_points * sizeof(MKL_Complex16), 64);
	this->psi_new = (MKL_Complex16*)mkl_malloc(sim_data.num_points * sizeof(MKL_Complex16), 64);
	this->conj_psi = (MKL_Complex16*)mkl_malloc(sim_data.num_points * sizeof(MKL_Complex16), 64);
	this->psi_temp = (MKL_Complex16*)mkl_malloc(sim_data.num_points * sizeof(MKL_Complex16), 64);
	this->psi_abs2 = (double*)mkl_malloc(sim_data.num_points * sizeof(double), 64);
	this->wavefunction_norm = 1;

	double expval;
	#pragma omp parallel for private(expval)
	for (int i = 0; i < sim_data.num_points; ++i) {
		expval = exp(-0.5 * pow(sim_data.x[i], 2.0));
		this->psi[i].real = expval;
		this->psi[i].imag = 0;
		this->psi_old[i].real = expval;
		this->psi_old[i].imag = 0;
		this->psi_new[i].real = 0;
		this->psi_new[i].imag = 0;
		this->conj_psi[i].real = 0;
		this->conj_psi[i].imag = 0;
		this->psi_abs2[i] = 0;
	}

	calc_norm(sim_data, this->psi);
	normalize_wf(sim_data, this->psi);
	vzAbs(sim_data.num_points, this->psi, this->psi_abs2);
	vdMul(sim_data.num_points, this->psi_abs2, this->psi_abs2, this->psi_abs2);
	save_data(this->psi_abs2, sim_data, "init_state.bin");

}

void WavefunctionData::calc_norm(SimulationData &sim_data, MKL_Complex16 *wf) {

	vzAbs(sim_data.num_points, wf, this->psi_abs2);
	vdMul(sim_data.num_points, this->psi_abs2, this->psi_abs2, this->psi_abs2);
	double sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < sim_data.num_points; ++i) {
		sum += this->psi_abs2[i];
	}

	this->wavefunction_norm = sum;
}

void WavefunctionData::normalize_wf(SimulationData &sim_data,  MKL_Complex16 *wf) {
	double temp_real = 0;
	double temp_imag = 0;
	#pragma omp parallel for private(temp_real, temp_imag)
	for (int i = 0; i < sim_data.num_points; ++i) {
		temp_real = wf[i].real * pow((this->wavefunction_norm * sim_data.dx), -0.5);
		temp_imag = wf[i].imag * pow((this->wavefunction_norm * sim_data.dx), -0.5);
		wf[i].real = temp_real;
		wf[i].imag = temp_imag;
	}

}

WavefunctionData::~WavefunctionData() {
	mkl_free(psi);
	mkl_free(psi_new);
	mkl_free(psi_old);
	mkl_free(psi_temp);
	mkl_free(psi_abs2);
	mkl_free(conj_psi);
}



