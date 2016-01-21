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

#include <stdlib.h>
#include <iostream>

#include "mkl.h"

#include "../include/simulation_data.hpp"
#include "../include/wavefunction_data.hpp"
#include "../include/potential_data.hpp"
#include "../include/solve.hpp"
#include "../include/save_data.hpp"

int main() {

	putenv("KMP_BLOCKTIME=infinite");
	putenv("KMP_AFFINITY=verbose,granularity=fine,compact,norespect");
	mkl_set_num_threads(mkl_get_max_threads());
	mkl_disable_fast_mm();
	
	SimulationData sim_data(2048);
	save_data(sim_data.x, sim_data, "x.bin");
	WavefunctionData wavefunction_data(sim_data);
	PotentialData potential_data(sim_data);
	
	vzAbs(sim_data.num_points, wavefunction_data.psi, wavefunction_data.psi_abs2);
	vdMul(sim_data.num_points, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2, wavefunction_data.psi_abs2);

	SolveImaginaryTime(sim_data, potential_data, wavefunction_data);

	SolveRealTime(sim_data, potential_data, wavefunction_data);

	return 0;
}
