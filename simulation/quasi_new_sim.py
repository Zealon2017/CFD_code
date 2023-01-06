import matplotlib.pyplot as plt
import taichi as ti
import numpy as np

ti.init(arch = ti.cpu, debug = False, kernel_profiler = True)

import LBM_2D_TP_MRT_Solver as lb2dtms

lb2d = lb2dtms.LB2D_Solver(nx = 477, ny = 171, nu_ratio = 5.0, theta = np.pi*1/6, sigma = 0.0135207)

lb2d.init_geo('./mat_model/porous_1.dat', './mat_model/porous_rho_1.dat')

lb2d.geo_process()

lb2d.set_periodic(fext = [0.0, 0.0])

# lb2d.set_velpre(vel = [1e-4, 0.0], rho = 1.0)

lb2d.init_simulation()

for iter in range(lb2d.N_max):
    
    lb2d.solve_mbf()

    if (iter % 1000 == 0):
        
        error = lb2d.Err()
        
        if (error < lb2d.tol):  
            lb2d.export_tecplot(iter)
            break
        
        if (iter % 5000 == 0):    
            lb2d.export_tecplot(iter)
        
       
# display the velocity distribution of the last step        
#temp = lb2d.convert_var()
#plt.imshow(temp)
ti.profiler.print_kernel_profiler_info()

            
