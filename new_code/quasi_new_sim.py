import taichi as ti
import numpy as np

#ti.init(arch = ti.cpu, default_fp = ti.f32, cpu_max_num_threads = 16, debug = False, kernel_profiler = False)

ti.init(arch = ti.gpu, kernel_profiler = True, debug = False)

import LBM_2D_CR_Solver as lb2dcrs

lb2d = lb2dcrs.LB2D_Solver(nx = 2013, ny = 1143, nu_ratio = 5.0, theta = np.pi*1/6, sigma = 0.015396)

lb2d.init_geo('./mat_model/porous_1.dat', './mat_model/porous_rho_1.dat')

lb2d.geo_process()

lb2d.set_periodic(fext = [0.0, 0.0])

lb2d.init_simulation()

for iter in range(lb2d.N_max):
    
    lb2d.solve_mbf()

    if (iter % 1000 == 0):
        
        error = lb2d.Err()
        
        if (error < lb2d.tol):  
            lb2d.export_tecplot(iter)
            break
        
        if (iter % 20000 == 0):    
            lb2d.export_tecplot(iter)
        
       
# display the velocity distribution of the last step        

#ti.profiler.print_kernel_profiler_info()

            
