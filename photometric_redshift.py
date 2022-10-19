''' This code provides a simple model to estimate galaxy redshift from its photometric data. It contains two functions. 

    The first function 'get_parameter' takes in the name of a training file of galaxies with known photometric data and redshifts, 
    to produce parameters for model 'z = b_0 + sum_k(b_k*m_k), where m_k are the magnitudes observed for different wavelength band.
    b_0 and b_k are the desired parameters. This function returns two items: a float b_0, and a list of float b_k. 
    
    The second function 'get_redshift' takes in a given set of paramters as well as a list of magnitudes in different band for a galaxy, 
    and estimate its redshift from the pre-calculated model. 
    This function takes in three items as parameters: a float b0, a list of float b_k, and a list of magnitudes. 
    It returns a single float which is the estimated redshift.
'''
def get_parameters(file_name):
    import numpy as np
    from astropy.io import ascii
    
    
    ztrain = ascii.read(file_name)
    
    #u, g, r, i, z stands for ultraviolet, green, red, near infrared and infrared band magnitudes, following SDSS standard. 
    [z, mu, mg, mr, mi, mz] = [ztrain["z"], ztrain["modelMag_u"], ztrain["modelMag_g"], 
                        ztrain["modelMag_r"], ztrain["modelMag_i"], ztrain["modelMag_z"]]
    mlist = [mu, mg, mr, mi, mz]
    
    
    #This is a function for calculating the covariance between two variables. 
    def cov(x, y):
        mx = np.mean(x)
        my = np.mean(y)
        n = len(x)
        s = 0
        for i in range(n):
            s += (x[i]-mx) * (y[i] - my)
        return s/n
    
    #This part calculates the covariance between redshift and each band of magnitudes.
    covzm = []
    for mag in mlist:
        covzm.append(cov(z, mag))
        
    #This part calculates the covariance between magnitudes and magnitudes.    
    mtxmm = []
    for i in mlist:
        row = []
        for j in mlist:
            row.append(cov(i,j))
        mtxmm.append(row)
        
    #The b_k parameters can be easily retrieved by solving the linear equation.    
    b_list = np.linalg.solve(mtxmm, covzm)

    #b_0 is just mean of redshift minus the sum of b_k times mean of m_k.
    sumbm = 0
    for i in range(5):
        sumbm += b_list[i] * np.mean(mlist[i])
    b0 = np.mean(z) - sumbm
    
    return b0, b_list

def get_redshift(b0, b_list, mag_list):
    return b0 + np.dot(b_list, mag_list)

