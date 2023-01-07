# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:54:49 2022

@author: zhilin
"""

import taichi as ti
import numpy as np

@ti.data_oriented
class LB2D_Solver:
    def __init__(self, nx, ny, nu_ratio, theta, sigma, sparse_storage = False):
        self.enable_projection = True
        self.nx, self.ny = nx, ny
        self.nu_ratio = nu_ratio
        self.theta = theta
        self.sigma = sigma
        self.tol = 1e-8
        self.N_max = 100000001
        self.beta = 0.5
        self.dh = 8.0
        self.k_frac = self.dh*self.dh/12.0
        
        self.fx, self.fy = 0.0, 0.0 
        self.bc_in_vx = 0.0
        self.bc_in_vy = 0.0
        
        self.tau_b = 0.52
        self.tau_r = 0.5 + (self.tau_b - 0.5)*self.nu_ratio
        self.f = ti.Vector.field(9, ti.f32, shape = (nx, ny))
        self.f_r = ti.Vector.field(9, ti.f32, shape = (nx, ny))
        self.f_b = ti.Vector.field(9, ti.f32, shape = (nx, ny))

        self.F = ti.Vector.field(9, ti.f32, shape = (nx, ny))
        self.F_r = ti.Vector.field(9, ti.f32, shape = (nx, ny))
        self.F_b = ti.Vector.field(9, ti.f32, shape = (nx, ny))

        self.psi = ti.field(ti.f32, shape = (nx, ny))
        self.psi_old = ti.field(ti.f32, shape = (nx, ny))

        self.delta_psi = ti.Vector.field(2, ti.f32, shape = (nx, ny))        
        self.norm_psi = ti.field(ti.f32, shape = (nx, ny))
        
        self.N_delta_psi = ti.Vector.field(2, ti.f32, shape = (nx, ny)) 
        self.N_delta_psi_old = ti.Vector.field(2, ti.f32, shape = (nx, ny))

        self.N_delta_psi_xy = ti.Vector.field(4, ti.f32, shape = (nx, ny)) 
        self.kappa = ti.field(ti.f32, shape = (nx, ny))
        self.ift = ti.Vector.field(2, ti.f32, shape = (nx, ny)) 

        self.tau = ti.field(ti.f32, shape = (nx, ny)) 

        self.rho = ti.field(ti.f32, shape = (nx, ny))
        self.rho_r = ti.field(ti.f32, shape = (nx, ny))
        self.rho_b = ti.field(ti.f32, shape = (nx, ny))
        self.obst_node = ti.field(ti.i32, shape = (nx, ny))

        self.wall_vec = ti.Vector.field(2, ti.f32, shape = (nx,ny))

        self.v = ti.Vector.field(2, ti.f32, shape = (nx, ny))
        self.v0 = ti.Vector.field(2, ti.f32, shape = (nx, ny))
        self.v_in = ti.field(ti.f32, shape = (ny))

        self.e = ti.Vector.field(2, ti.i32, shape = (9))
        self.S_dig = ti.Matrix.field(9, 9, ti.f32, shape = ())
        self.w = ti.field(ti.f32, shape = (9))
        self.solid = ti.field(ti.i32,shape = (nx, ny))
        self.solid_rho = ti.field(ti.i32,shape = (nx, ny))
        self.M = ti.field(ti.f32, shape=(9, 9))
        self.inv_M = ti.field(ti.f32, shape=(9, 9))
        M_np = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [-4,-1,-1,-1,-1, 2, 2, 2, 2],
                        [4,-2,-2,-2,-2, 1, 1, 1, 1],
                        [0, 1, 0,-1, 0, 1,-1,-1, 1],
                        [0,-2, 0, 2, 0, 1,-1,-1, 1],
                        [0, 0, 1, 0,-1, 1, 1,-1,-1],
                        [0, 0,-2, 0, 2, 1, 1,-1,-1],
                        [0, 1,-1, 1,-1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1,-1, 1,-1]], dtype = np.float32)
        inv_M_np = np.linalg.inv(M_np)
        self.M.from_numpy(M_np)
        self.inv_M.from_numpy(inv_M_np)

        self.LR = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        self.inflow = [0, 2, 3, 4, 6, 7]
        self.x = np.linspace(0, nx, nx)
        self.y = np.linspace(0, ny, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def init_simulation(self):      
        self.ext_f = ti.Vector([self.fx, self.fy]) 
        ti.static(self.LR)    
        ti.static(self.inflow)       
        self.static_init()
        self.init()

    @ti.func
    def M_multiply(self, matrix):
        temp = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                temp[i] += self.M[i,j]*matrix[j]
        return temp
    
    @ti.func
    def inv_M_multiply(self, matrix):
        temp = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                temp[i] += self.inv_M[i,j]*matrix[j]
        return temp

               
    @ti.func
    def feq(self, k, rho_local, u):       
        eu = ti.cast(self.e[k], ti.f32).dot(u)
        #eu = self.e[k].dot(u)
        uv = u.dot(u)
        feqout = self.w[k]*rho_local*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*uv)
        return feqout

    @ti.func
    def meq_c(self, rho_local, u):
        out = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out[0] = rho_local
        out[1] = rho_local*(3.0*u.dot(u) - 2.0)
        out[2] = rho_local*(1.0 - 3.0*u.dot(u))
        out[3] = rho_local*(u.x)
        out[4] = -rho_local*(u.x)
        out[5] = rho_local*(u.y)
        out[6] = -rho_local*(u.y)
        out[7] = rho_local*(u.x*u.x - u.y*u.y)
        out[8] = rho_local*(u.x*u.y)
        return out
    
    @ti.func
    def meq_guo_c(self, force, u):
        out = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out[0] = 0.0
        out[1] = 6.0*(u.x*force.x + u.y*force.y)
        out[2] = -6.0*(u.x*force.x + u.y*force.y)
        out[3] = force.x
        out[4] = -force.x
        out[5] = force.y
        out[6] = -force.y
        out[7] = 2.0*(u.x*force.x - u.y*force.y)
        out[8] = u.x*force.y + u.y*force.x
        return out              

    def init_geo(self, filename1, filename2):
        in_dat1 = np.loadtxt(filename1, dtype = np.int32)
        in_dat1 = np.reshape(in_dat1, (self.nx,self.ny), order='F')
        in_dat2 = np.loadtxt(filename2, dtype = np.int32)
        in_dat2 = np.reshape(in_dat2, (self.nx,self.ny), order='F')
        self.solid.from_numpy(in_dat1)
        self.solid_rho.from_numpy(in_dat2)

    def geo_process(self):
        temp_obst = self.Create_obst_node()
        self.obst_node.from_numpy(temp_obst)
        wall_vec_x, wall_vec_y = self.get_norm_wall()
        wall_temp = np.zeros((self.nx, self.ny, 2), dtype = np.float32)
        wall_temp[:,:,0] = wall_vec_x
        wall_temp[:,:,1] = wall_vec_y
        self.wall_vec.from_numpy(wall_temp)

    @ti.kernel
    def init(self):
        for i, j in self.solid:
            if (self.solid_rho[i,j] == 1):
                self.rho_b[i,j] = 1.0
                self.rho_r[i,j] = 0.0
            elif (self.solid_rho[i,j] == 0 and self.solid[i,j] == 0):
                self.rho_r[i,j] = 1.0
                self.rho_b[i,j] = 0.0
            else:
                self.rho_r[i,j] = -3.0
                self.rho_b[i,j] =  1.0    
                        
            self.psi[i,j] = (self.rho_r[i,j] - self.rho_b[i,j]) / (self.rho_r[i,j] + self.rho_b[i,j])
            self.psi_old[i,j] = self.psi[i,j]

            self.rho[i,j] = self.rho_r[i,j] + self.rho_b[i,j]
            self.tau[i,j] = 1.0
            self.v[i,j] = ti.Vector([0.0, 0.0])
            self.v0[i,j] = ti.Vector([0.0, 0.0])            

            self.delta_psi[i,j] = ti.Vector([0.0, 0.0])     
            self.norm_psi[i,j] = 1e-10        
            self.N_delta_psi[i,j] = ti.Vector([0.0, 0.0])
            self.N_delta_psi_old[i,j] = ti.Vector([0.0, 0.0])

            self.N_delta_psi_xy[i,j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.kappa[i,j] = 0.0
            self.ift[i,j] = ti.Vector([0.0, 0.0]) 

            for s in ti.static(range(9)):                   
                self.f_r[i,j][s] = self.feq(s, self.rho_r[i,j], self.v[i,j])
                self.f_b[i,j][s] = self.feq(s, self.rho_b[i,j], self.v[i,j])
                self.f[i,j][s] = self.f_r[i,j][s] + self.f_b[i,j][s]
                self.F_r[i,j][s] = self.f_r[i,j][s]
                self.F_b[i,j][s] = self.f_b[i,j][s]
                self.F[i,j][s] = self.f[i,j][s]  

    @ti.kernel
    def static_init(self):
        if ti.static(self.enable_projection): # No runtime overhead
            self.e[0] = ti.Vector([0,0])
            self.e[1] = ti.Vector([1,0]);   self.e[2] = ti.Vector([0,1])
            self.e[3] = ti.Vector([-1,0]);  self.e[4] = ti.Vector([0,-1])
            self.e[5] = ti.Vector([1,1]);   self.e[6] = ti.Vector([-1,1])
            self.e[7] = ti.Vector([-1,-1]); self.e[8] = ti.Vector([1,-1])
            
            self.w[0] = 4.0/9.0 
            self.w[1] = 1.0/9.0;  self.w[2] = 1.0/9.0
            self.w[3] = 1.0/9.0;  self.w[4] = 1.0/9.0
            self.w[5] = 1.0/36.0; self.w[6] = 1.0/36.0
            self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0
                       
    @ti.func
    def periodic_index(self,i):
        iout = i
        if i[0] < 0:     iout[0] = self.nx-1
        if i[0] > self.nx-1:  iout[0] = 0
        if i[1] < 0:     iout[1] = self.ny-1
        if i[1] > self.ny-1:  iout[1] = 0
        return iout

    @ti.kernel
    def streaming(self): # actually, this step involves the bounce-back step 
        for i in ti.grouped(self.rho):
            if (self.solid[i] == 0):
                for k in ti.static(range(9)):
                    ip = self.periodic_index(i + self.e[k])
                    if (self.solid[ip] == 0):
                        self.F_r[ip][k] = self.f_r[i][k]
                        self.F_b[ip][k] = self.f_b[i][k]
                    else:
                        self.F_r[i][self.LR[k]] = self.f_r[i][k]
                        self.F_b[i][self.LR[k]] = self.f_b[i][k]

    @ti.kernel
    def BC_periodic_mod1(self): 
        for j in range(1, self.ny-1): 
            # inlet treatment 
            self.F_r[0,j][1] = self.f_b[self.nx-1,j][1]
            self.F_b[0,j][1] = self.f_r[self.nx-1,j][1]

            self.F_r[0,j][5] = self.f_b[self.nx-1,j-1][5]
            self.F_b[0,j][5] = self.f_r[self.nx-1,j-1][5]

            self.F_r[0,j][8] = self.f_b[self.nx-1,j+1][8]
            self.F_b[0,j][8] = self.f_r[self.nx-1,j+1][8]

            # outlet treatment 
            self.F_r[self.nx-1,j][3] = self.f_b[0,j][3]
            self.F_b[self.nx-1,j][3] = self.f_r[0,j][3]

            self.F_r[self.nx-1,j][6] = self.f_b[0,j-1][6]
            self.F_b[self.nx-1,j][6] = self.f_r[0,j-1][6]

            self.F_r[self.nx-1,j][7] = self.f_b[0,j+1][7]
            self.F_b[self.nx-1,j][7] = self.f_r[0,j+1][7]

        self.F_r[0,0][1] = self.f_b[self.nx-1,0][1]
        self.F_b[0,0][1] = self.f_r[self.nx-1,0][1]
        self.F_r[0,0][5] = self.f_b[self.nx-1,self.ny-1][5]
        self.F_b[0,0][5] = self.f_r[self.nx-1,self.ny-1][5]
        self.F_r[0,0][8] = self.f_b[self.nx-1,1][8]
        self.F_b[0,0][8] = self.f_r[self.nx-1,1][8]

        self.F_r[0,self.ny-1][1] = self.f_b[self.nx-1,self.ny-1][1]
        self.F_b[0,self.ny-1][1] = self.f_r[self.nx-1,self.ny-1][1]
        self.F_r[0,self.ny-1][5] = self.f_b[self.nx-1,self.ny-2][5]
        self.F_b[0,self.ny-1][5] = self.f_r[self.nx-1,self.ny-2][5]
        self.F_r[0,self.ny-1][8] = self.f_b[self.nx-1,0][8]
        self.F_b[0,self.ny-1][8] = self.f_r[self.nx-1,0][8]

        self.F_r[self.nx-1,0][3] = self.f_b[0,0][3]
        self.F_b[self.nx-1,0][3] = self.f_r[0,0][3]
        self.F_r[self.nx-1,0][6] = self.f_b[0,self.ny-1][6]
        self.F_b[self.nx-1,0][6] = self.f_r[0,self.ny-1][6]
        self.F_r[self.nx-1,0][7] = self.f_b[0,1][7]
        self.F_b[self.nx-1,0][7] = self.f_r[0,1][7]

        self.F_r[self.nx-1,self.ny-1][3] = self.f_b[0,self.ny-1][3]
        self.F_b[self.nx-1,self.ny-1][3] = self.f_r[0,self.ny-1][3]
        self.F_r[self.nx-1,self.ny-1][6] = self.f_b[0,self.ny-2][6]
        self.F_b[self.nx-1,self.ny-1][6] = self.f_r[0,self.ny-2][6]
        self.F_r[self.nx-1,self.ny-1][7] = self.f_b[0,0][7]
        self.F_b[self.nx-1,self.ny-1][7] = self.f_r[0,0][7]

    def solve_mbf(self):
        self.macros_psi_fourth()
        self.modify_cfb()
        self.get_Npsi_ift_col_recoloring()
        self.streaming()
        self.BC_periodic_mod1() # only for validation 
        #self.BC_periodic_mod2()
        self.copy_last_variables()
   
    def convert_var(self):
        # vel = self.v.to_numpy()
        # vel_mag = (vel[:, :, 0]**2.0 + vel[:, :, 1]**2.0)**0.5
        # vel_mag = vel[:, :, 1]
        # return vel_mag
        psi = self.psi.to_numpy()
        return psi

    @ti.kernel
    def Err(self)-> ti.f32:
        error = ti.Vector([0.0, 0.0])
        for i in ti.grouped(self.rho):
            if (self.solid[i] == 0):
                error[0] += ti.sqrt(self.v[i].norm_sqr())
                error[1] += ti.sqrt(self.v0[i].norm_sqr())       
                self.v0[i] = self.v[i]
        return ti.abs(error[0] / error[1] - 1.0) 

    def export_tecplot(self, n):
        temp_solid = self.solid.to_numpy()
        temp_rho =self.rho.to_numpy()
        temp_psi =self.psi.to_numpy()
        VX = self.v.to_numpy()[0:self.nx, 0:self.ny, 0]
        VY = self.v.to_numpy()[0:self.nx, 0:self.ny, 1]
        filename = "./output/LB_"+str(n)+".dat"
        with open(filename, 'w') as f:
            f.write('Variables = "X", "Y", "obst", "rho", "psi", "VX", "VY"\n')
            f.write('Zone I='+str(self.nx)+', J='+str(self.ny)+', F=POINT\n')
            for j in range(self.ny):
                for i in range(self.nx):
                    f.write(str(self.X[i,j]) + ' ' + str(self.Y[i,j])+ ' ' + str(temp_solid[i,j])+ ' ' \
                    + str(temp_rho[i,j])+ ' ' +str(temp_psi[i,j])+ ' ' +str(VX[i,j])+ ' ' + str(VY[i,j]))
                    f.write('\n')
            f.close()

    # Here, I will add some new functions
    def Create_obst_node(self):
        cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
        cy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
        solid_dat = self.solid.to_numpy()
        dnx = solid_dat.shape[0]
        dny = solid_dat.shape[1]
        obst_node = np.zeros((dnx, dny), dtype = np.int32)
        top = solid_dat[0,:].reshape(1,dny)
        bottom = solid_dat[-1,:].reshape(1,dny)
        new1 = np.concatenate((top, solid_dat), axis = 0) # row
        new2 = np.concatenate((new1, bottom), axis = 0) # row
        left = new2[:,0].reshape(dnx+2,1)
        right = new2[:,-1].reshape(dnx+2,1)
        new3 = np.concatenate((left, new2), axis = 1) # column
        new4 = np.concatenate((new3, right), axis = 1) # column
        N_dnx = new4.shape[0]
        N_dny = new4.shape[1]    

        for i in range(1, N_dnx-1):
            for j in range(1, N_dny-1):
                value = 0
                for k in range(9):
                    coor_x = i + cx[k]
                    coor_y = j + cy[k]
                    value += (1 - new4[coor_x,coor_y]) 
                if (new4[i,j] == 1 and value == 0):
                    obst_node[i-1,j-1] = 4
                elif (new4[i,j] == 1 and value > 0):
                    obst_node[i-1,j-1] = 3
                elif (new4[i,j] == 0 and value == 9):
                    obst_node[i-1,j-1] = 2
                elif (new4[i,j] == 0 and value < 9):
                    obst_node[i-1,j-1] = 1
        return obst_node

    def get_norm_wall(self):
        solid_dat = self.solid.to_numpy()
        obst_node = self.obst_node.to_numpy()
        dnx = solid_dat.shape[0]
        dny = solid_dat.shape[1]
        wall_vec_x = np.zeros((dnx, dny), dtype = np.float32)
        wall_vec_y = np.zeros((dnx, dny), dtype = np.float32)
        top = solid_dat[0,:].reshape(1,dny)    
        bottom = solid_dat[-1,:].reshape(1,dny)
        new1 = np.concatenate((top, top, solid_dat), axis = 0) # row
        new2 = np.concatenate((new1, bottom, bottom), axis = 0) # row
        left = new2[:,0].reshape(dnx + 4, 1)
        right = new2[:,-1].reshape(dnx + 4, 1)
        new3 = np.concatenate((left, left, new2), axis = 1) # column
        obst_tar = np.concatenate((new3, right, right), axis = 1) # column 
        cxx = [0, 1, 0, -1, 0, 1, -1, -1, 1, 2, 0, -2, 0, 2, -2, -2, 2, 2, 1, -1, -2, -2, -1, 1, 2]
        cyy = [0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2, 2, 2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1]
        ww = np.zeros(25) 
        ww[1:5] = 4/21; ww[5:9] = 4/45; ww[9:13] = 1/60; ww[13:17] = 1/5040; ww[17:25] = 2/315       
        for ix in range(dnx):
            for jy in range(dny):
                if (obst_node[ix, jy] == 1):
                    i = ix + 2
                    j = jy + 2
                    temp_x = 0.0
                    temp_y = 0.0
                    for k in range(25):
                        coor_x = i + cxx[k]
                        coor_y = j + cyy[k]
                        temp_x += (obst_tar[coor_x,coor_y]*ww[k]*cxx[k])
                        temp_y += (obst_tar[coor_x,coor_y]*ww[k]*cyy[k])
                    norm_t = np.sqrt(temp_x**2.0 + temp_y**2.0) + 1e-15
                    wall_vec_x[ix, jy] = temp_x / norm_t
                    wall_vec_y[ix, jy] = temp_y / norm_t
                else:
                    wall_vec_x[ix, jy] = 0.0
                    wall_vec_y[ix, jy] = 0.0                                
        return wall_vec_x, wall_vec_y

    @ti.func
    def get_psi_csb(self, i):
        temp_psi = 0.0
        temp_w = 1e-15
        for k in ti.static(range(9)):
            coor_x = i[0] + self.e[k].x 
            coor_y = i[1] + self.e[k].y 
            if (coor_x >= 0 and coor_y >= 0 and coor_x < self.nx and coor_y < self.ny and self.obst_node[coor_x,coor_y] == 1):
                temp_psi += self.w[k]*self.psi_old[coor_x, coor_y]
                temp_w += self.w[k]
        return temp_psi/temp_w

    @ti.func
    def get_yu(self, i, j):
        sum_w = 1e-10
        nx_w = 0.0
        ny_w = 0.0
        for k in ti.static(range(9)):
            coor_x = i + self.e[k].x 
            coor_y = j + self.e[k].y 
            if (coor_x >= 0 and coor_y >= 0 and coor_x < self.nx and coor_y < self.ny and self.obst_node[coor_x,coor_y] == 1):
                nx_w += self.w[k]*self.N_delta_psi_old[coor_x, coor_y][0]
                ny_w += self.w[k]*self.N_delta_psi_old[coor_x, coor_y][1]
                sum_w += self.w[k]
        return nx_w/sum_w, ny_w/sum_w

    @ti.kernel
    def copy_last_variables(self):
        for i,j in self.solid:
            if (self.obst_node[i,j] == 1):
                self.psi_old[i,j] = self.psi[i,j]
                self.N_delta_psi_old[i,j] = self.N_delta_psi[i,j]

    @ti.kernel    
    def modify_cfb(self):
        #ti.loop_config(serialize = True)
        for i,j in self.solid:
            if (self.obst_node[i,j] == 1):
                theta_temp = ti.acos(self.wall_vec[i,j][0]*self.N_delta_psi[i,j][0] + self.wall_vec[i,j][1]*self.N_delta_psi[i,j][1])
                temp1 = ti.cos(self.theta) - ti.sin(self.theta) * ti.cos(theta_temp) / ti.sin(theta_temp)
                temp2 = ti.sin(self.theta) / ti.sin(theta_temp)
                npx = temp1*self.wall_vec[i,j][0] + temp2*self.N_delta_psi[i,j][0]
                npy = temp1*self.wall_vec[i,j][1] + temp2*self.N_delta_psi[i,j][1]
                temp3 = ti.cos(-self.theta) - ti.sin(-self.theta) * ti.cos(theta_temp) / ti.sin(theta_temp)
                temp4 = ti.sin(-self.theta) / ti.sin(theta_temp)
                nmx = temp3*self.wall_vec[i,j][0] + temp4*self.N_delta_psi[i,j][0]
                nmy = temp3*self.wall_vec[i,j][1] + temp4*self.N_delta_psi[i,j][1]
                D1 = ti.sqrt((npx - self.N_delta_psi[i,j][0])**2.0 + (npy - self.N_delta_psi[i,j][1])**2.0)
                D2 = ti.sqrt((nmx - self.N_delta_psi[i,j][0])**2.0 + (nmy - self.N_delta_psi[i,j][1])**2.0)
                if (D1 < D2):
                    self.N_delta_psi[i,j][0] = npx
                    self.N_delta_psi[i,j][1] = npy
                elif (D1 > D2):
                    self.N_delta_psi[i,j][0] = nmx
                    self.N_delta_psi[i,j][1] = nmy  
                else:
                    self.N_delta_psi[i,j][0] = self.wall_vec[i,j][0]
                    self.N_delta_psi[i,j][1] = self.wall_vec[i,j][1]   
                self.delta_psi[i,j][0] = -self.norm_psi[i,j]*self.N_delta_psi[i,j][0]
                self.delta_psi[i,j][1] = -self.norm_psi[i,j]*self.N_delta_psi[i,j][1]
            elif (self.obst_node[i,j] == 3):
                self.N_delta_psi[i,j][0],self.N_delta_psi[i,j][1] = self.get_yu(i,j)

    @ti.kernel
    def get_Npsi_ift_col_recoloring(self):
        for i in ti.grouped(self.rho):
            if (self.solid[i] == 0):
                # get N_delta_psi_xy
                self.N_delta_psi_xy[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                if (i[0] == 0):
                    tx = i[0]
                    ty = i[1]
                    self.N_delta_psi_xy[i][0], self.N_delta_psi_xy[i][1] = self.in_N_delta_psi_xy12(tx, ty)
                    self.N_delta_psi_xy[i][2], self.N_delta_psi_xy[i][3] = self.in_N_delta_psi_xy34(tx, ty)
                elif (i[0] == (self.ny-1)):
                    tx = i[0]
                    ty = i[1]
                    self.N_delta_psi_xy[i][0], self.N_delta_psi_xy[i][1] = self.out_N_delta_psixy12(tx, ty)
                    self.N_delta_psi_xy[i][2], self.N_delta_psi_xy[i][3] = self.out_N_delta_psixy34(tx, ty)   
                else:                                      
                    for k in ti.static(range(9)):
                        ip = self.periodic_index(i + self.e[k])                    
                        self.N_delta_psi_xy[i][0] += 3.0*ti.cast(self.e[k].x,ti.f32)*(self.w[k]*self.N_delta_psi[ip][0])
                        self.N_delta_psi_xy[i][1] += 3.0*ti.cast(self.e[k].y,ti.f32)*(self.w[k]*self.N_delta_psi[ip][0])            
                        self.N_delta_psi_xy[i][2] += 3.0*ti.cast(self.e[k].x,ti.f32)*(self.w[k]*self.N_delta_psi[ip][1])
                        self.N_delta_psi_xy[i][3] += 3.0*ti.cast(self.e[k].y,ti.f32)*(self.w[k]*self.N_delta_psi[ip][1])  

                # get kappa and ift 
                self.kappa[i] = self.N_delta_psi[i][0]*self.N_delta_psi[i][1]*(self.N_delta_psi_xy[i][1] + self.N_delta_psi_xy[i][2])-\
                    (self.N_delta_psi[i][1]**2.0*self.N_delta_psi_xy[i][0] + self.N_delta_psi[i][0]**2.0*self.N_delta_psi_xy[i][3]) - 4.0*ti.cos(self.theta)/self.dh
                self.ift[i] = -0.5*self.sigma*self.kappa[i]*self.delta_psi[i]
                # collision step 
                self.F[i] = self.F_r[i] + self.F_b[i]
                
                m_f = self.M_multiply(self.F[i])
                m_feq = self.meq_c(self.rho[i], self.v[i])                              
                self.tau[i] = 0.5 + 1.0/((1.0 + self.psi[i])/(2.0*self.tau_r - 1.0) + (1.0 - self.psi[i])/(2.0*self.tau_b - 1.0))
                temp_vis = 1.0/3.0*self.rho[i]*(self.tau[i]-0.5)
                force_meq = self.meq_guo_c(self.ext_f + self.ift[i] - temp_vis/self.k_frac*self.v[i], self.v[i]) 
                dig_S = ti.Vector([1.0, 1.64, 1.54, 1.0, 1.7, 1.0, 1.7, 1.0/self.tau[i], 1.0/self.tau[i]])
                diag_F = 1.0 - dig_S/2.0
                m_f -= (dig_S*(m_f - m_feq) - diag_F*force_meq)
                self.f[i] = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                self.f[i] += self.inv_M_multiply(m_f)
                
                # recoloring step 
                temp1 = self.rho_r[i] / self.rho[i]
                temp2 = self.rho_b[i] / self.rho[i]
                cosfai = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                for kk in ti.static(range(9)):               
                    cosfai[kk] = ti.cast(self.e[kk],ti.f32).dot(self.delta_psi[i])/self.norm_psi[i]
                    self.f_r[i][kk] = temp1*self.f[i][kk] + self.beta*temp1*temp2*cosfai[kk]*self.w[kk]
                    self.f_b[i][kk] = temp2*self.f[i][kk] - self.beta*temp1*temp2*cosfai[kk]*self.w[kk]

    @ti.kernel
    def macros_psi_fourth(self):
        for i in ti.grouped(self.rho):
            if (self.obst_node[i] <= 2):
                # get velocity and density
                self.rho_r[i] = 0.0
                self.rho_b[i] = 0.0                
                self.rho_r[i] += self.F_r[i].sum()
                self.rho_b[i] += self.F_b[i].sum()
                self.rho[i] = self.rho_r[i] + self.rho_b[i]                
                self.v[i] = ti.Vector([0.0, 0.0])
                for s in ti.static(range(9)):
                    self.v[i] += ti.cast(self.e[s],ti.f32)*(self.F_r[i][s] + self.F_b[i][s]) 
                temp_vis = 1.0/3.0*self.rho[i]*(self.tau[i] - 0.5) 
                self.v[i] += 0.5*(self.ext_f + self.ift[i])
                self.v[i] /= (self.rho[i] + 0.5*temp_vis/self.k_frac)
                
                # get psi and its fourth order value 
                self.psi[i] = (self.rho_r[i] - self.rho_b[i]) / (self.rho_r[i] + self.rho_b[i])
                self.delta_psi[i] = ti.Vector([0.0, 0.0])
                self.N_delta_psi[i] = ti.Vector([0.0, 0.0])

                if (i[0] == 0):
                    tx = i[0]
                    ty = i[1]
                    self.delta_psi[i][0], self.delta_psi[i][1] = self.in_delta_psi(tx,ty)    
                elif (i[0] == (self.ny-1)):
                    tx = i[0]
                    ty = i[1]                
                    self.delta_psi[i][0], self.delta_psi[i][1] = self.out_delta_psi(tx,ty)  
                else:
                    for k in ti.static(range(9)):
                        ip = self.periodic_index(i + self.e[k])                   
                        self.delta_psi[i] += 3.0*ti.cast(self.e[k],ti.f32)*(self.w[k]*self.psi[ip])
                self.norm_psi[i] = ti.sqrt(self.delta_psi[i].norm_sqr()) + 1e-15                
                self.N_delta_psi[i] = -self.delta_psi[i] / self.norm_psi[i]
            elif (self.obst_node[i] == 3): # here is the solid node
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0.0,0.0])
                self.psi[i] = self.get_psi_csb(i)
            else:
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0.0,0.0])
                self.psi[i] = (self.rho_r[i] - self.rho_b[i]) / (self.rho_r[i] + self.rho_b[i])
    
    @ti.func
    def in_delta_psi(self, i, j): 
        ip = i + 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1
            
        temp0 = self.psi[i,j]; temp1 = self.psi[ip,j]; temp2 = self.psi[i,jp]
        temp4 = self.psi[i,jb]; temp5 = self.psi[ip,jp]; temp8 = self.psi[ip,jb]
        temp3 = 2*temp0 - temp1
        temp6 = 2*temp2 - temp5
        temp7 = 2*temp4 - temp8 
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])
        return tempx, tempy
    
    @ti.func
    def out_delta_psi(self, i, j): 
        ib = i - 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1
        
        temp0 = self.psi[i,j]; temp2 = self.psi[i,jp]; temp3 = self.psi[ib,j]
        temp4 = self.psi[i,jb]; temp6 = self.psi[ib,jp]; temp7 = self.psi[ib,jb]
        temp1 = 2*temp0 - temp3
        temp5 = 2*temp2 - temp6
        temp8 = 2*temp4 - temp7
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0  
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])            
        return tempx, tempy    
         
    @ti.func
    def in_N_delta_psi_xy12(self, i, j): 
        ip = i + 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1

        temp0 = self.N_delta_psi[i,j][0]; temp1 = self.N_delta_psi[ip,j][0]; temp2 = self.N_delta_psi[i,jp][0]
        temp4 = self.N_delta_psi[i,jb][0]; temp5 = self.N_delta_psi[ip,jp][0]; temp8 = self.N_delta_psi[ip,jb][0]
        temp3 = 2*temp0 - temp1
        temp6 = 2*temp2 - temp5
        temp7 = 2*temp4 - temp8 
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])
        return tempx, tempy

    @ti.func
    def in_N_delta_psi_xy34(self, i, j): 
        ip = i + 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1
            
        temp0 = self.N_delta_psi[i,j][1]; temp1 = self.N_delta_psi[ip,j][1]; temp2 = self.N_delta_psi[i,jp][1]
        temp4 = self.N_delta_psi[i,jb][1]; temp5 = self.N_delta_psi[ip,jp][1]; temp8 = self.N_delta_psi[ip,jb][1]
        temp3 = 2*temp0 - temp1
        temp6 = 2*temp2 - temp5
        temp7 = 2*temp4 - temp8 
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])
        return tempx, tempy
    
    @ti.func
    def out_N_delta_psixy12(self, i, j): 
        ib = i - 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1

        temp0 = self.N_delta_psi[i,j][0]; temp2 = self.N_delta_psi[i,jp][0]; temp3 = self.N_delta_psi[ib,j][0]
        temp4 = self.N_delta_psi[i,jb][0]; temp6 = self.N_delta_psi[ib,jp][0]; temp7 = self.N_delta_psi[ib,jb][0]
        temp1 = 2*temp0 - temp3
        temp5 = 2*temp2 - temp6
        temp8 = 2*temp4 - temp7
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0  
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])            
        return tempx, tempy  

    @ti.func
    def out_N_delta_psixy34(self, i, j): 
        ib = i - 1
        
        jp = j + 1
        if (jp > self.ny-1):
            jp = 0
        
        jb = j - 1
        if (jb < 0):
            jb = self.ny-1

        temp0 = self.N_delta_psi[i,j][1]; temp2 = self.N_delta_psi[i,jp][1]; temp3 = self.N_delta_psi[ib,j][1]
        temp4 = self.N_delta_psi[i,jb][1]; temp6 = self.N_delta_psi[ib,jp][1]; temp7 = self.N_delta_psi[ib,jb][1]
        temp1 = 2*temp0 - temp3
        temp5 = 2*temp2 - temp6
        temp8 = 2*temp4 - temp7
        temp = ti.Vector([temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8])
        tempx = 0.0
        tempy = 0.0  
        for k in ti.static(range(9)):
            tempx += (3.0*self.w[k]*ti.cast(self.e[k][0],ti.f32)*temp[k])
            tempy += (3.0*self.w[k]*ti.cast(self.e[k][1],ti.f32)*temp[k])            
        return tempx, tempy

    def set_periodic(self, fext):
        self.fx = fext[0]
        self.fy = fext[1] 







                                                  




                
                





            









        





        
        

        
        
        
        