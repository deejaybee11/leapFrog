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

#include "../include/solve.hpp"

#include <stdlib.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "mkl.h"

#include "../include/simulation_data.hpp"
#include "../include/wavefunction_data.hpp"
#include "../include/potential_data.hpp"
#include "../include/save_data.hpp"

void SolveImaginaryTime(SimulationData &sim_data, PotentialData &potential_data, WavefunctionData &wavefunction_data) {

	int i = 0;
	bool BREAK = false;	
	wavefunction_data.calc_norm(sim_data, wavefunction_data.psi);
	while (!BREAK) {

		if (i % 100000 == 0) {
			std::cout << "Step = " << i << std::endl;
		}

		if((wavefunction_data.get_norm() * sim_data.dx > 2) || (wavefunction_data.get_norm() * sim_data.dx < 0.5)) {
			std::cout << "NORM TOO LARGE = " << wavefunction_data.get_norm() * sim_data.dx <<  " - ITERATION " << i  << std::endl;
			BREAK = true;	
		}
		wavefunction_data.psi[0].real = 0;
		wavefunction_data.psi[sim_data.num_points - 1].real = 0;	
		vzAbs(sim_data.num_points, wavefunction_data.psi, wavefunction_data.psi_abs2);
		vdMul(sim_data.num_points, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2);

		#pragma omp parallel for 
		for (int j = 1; j < sim_data.num_points - 1; ++j) {
			wavefunction_data.psi_new[j].real = wavefunction_data.psi[j].real - 2 * sim_data.dt * (sim_data.beta * wavefunction_data.psi_abs2[j] * wavefunction_data.psi[j].real + wavefunction_data.psi[j].real * potential_data.harmonic_trap[j] - 0.5 * (wavefunction_data.psi[j+1].real + wavefunction_data.psi[j-1].real - 2 * wavefunction_data.psi[j].real) / sim_data.dx2);
		}

		wavefunction_data.calc_norm(sim_data, wavefunction_data.psi_new);
		wavefunction_data.normalize_wf(sim_data, wavefunction_data.psi_new);

		#pragma omp parallel for
		for (int j = 0; j < sim_data.num_points; ++j) {
			wavefunction_data.psi[j].real = wavefunction_data.psi_new[j].real;
		}

		vzConj(sim_data.num_points, wavefunction_data.psi, wavefunction_data.conj_psi);
		vzAbs(sim_data.num_points, wavefunction_data.psi, wavefunction_data.psi_abs2);
		vdMul(sim_data.num_points, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2);

		double tempval;
		#pragma omp parallel for reduction(+:tempval)
		for (int j = 1; j < sim_data.num_points - 1; ++j) {
			tempval += wavefunction_data.conj_psi[j].real * (sim_data.beta * wavefunction_data.psi_abs2[j] * wavefunction_data.psi[j].real + wavefunction_data.psi[j].real * potential_data.harmonic_trap[j] - 0.5 * (wavefunction_data.psi[j+1].real + wavefunction_data.psi[j-1].real - 2 * wavefunction_data.psi[j].real) / sim_data.dx2) * wavefunction_data.psi[j].real;
		}
		sim_data.chemical_potential[i%2] = tempval;

		i += 1;
		if ((i > 20000) && (abs(sim_data.chemical_potential[1] - sim_data.chemical_potential[0])) < 1e-3) {
			std::cout << "GROUND STATE FOUND" << std::endl;
			BREAK = true;
		}
	}
	save_data(wavefunction_data.psi_abs2, sim_data, "ground_state.bin");
}
