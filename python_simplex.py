import numpy as np
# import subprocess
import os
import math 
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

Nfeval = 1

def cost_function(x):
    global Nfeval
    f = open("min-hist.txt",'a')

    text_alpha_C11 = 'alpha_C11 '+str(x[0])
    text_beta_C11 = 'beta_C11 '+str(x[1])
    text_alpha_C22 = 'alpha_C22 '+str(x[2])
    text_beta_C22 = 'beta_C22 '+str(x[3])
    #text_alpha_C33 = 'alpha_C33 '+str(x[4])
    #text_beta_C33 = 'beta_C33 '+str(x[5])
    os.system("sed -i '1s/.*/"+text_alpha_C11+"/' sample.in")
    os.system("sed -i '2s/.*/"+text_beta_C11+"/' sample.in")
    os.system("sed -i '3s/.*/"+text_alpha_C22+"/' sample.in")
    os.system("sed -i '4s/.*/"+text_beta_C22+"/' sample.in")
    #os.system("sed -i '5s/.*/"+text_alpha_C33+"/' sample.in")
    #os.system("sed -i '6s/.*/"+text_beta_C33+"/' sample.in")
    os.system("srun -n343 /g/g20/iyer7/0_CHARGE_DENSITY_FITTING/binaries/parallel/main sample.in >log")
              
    dataout = np.loadtxt("final_RMSE.txt")
    f.write("%5d %15.8f %12.8f %12.8f %12.8f %12.8f\n" % (Nfeval,dataout,x[0],x[1],x[2],x[3]))
    f.close()
    Nfeval +=1

    return dataout

x0 = [0.34629774874184366, 1.382352361467073, 0.22117137572101103, 1.3884000752515062]

res = minimize(cost_function, x0, method='nelder-mead', options={'xatol': 1e-6,'maxiter':10000, 'disp':True})

evaluate=OptimizeResult(res)

