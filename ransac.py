import numpy as np
import scipy as sp
import scipy.linalg as sl

def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    """
    Inputs:
		data    - sample points
		model   - hypothetical model
		n       - the least sample points needed to generate the model
		k       - maximal iterations
		t       - threshold for deciding the point satisfies the model or not
		d       - the least number of sample points when the fit is good.
		
	Outputs:
		best_fit - the optimal fitting solution
		        - return nil if not found
    
	Pseudocode:
	
    iterations = 0
    best_fit = nil                        # update later
    best_err = something really large     # update later that best_err = this_err
    while iterations < k 
    {
        possible_inliers = randomly select n sample points
        possible_model = n possible_inliers 
        also_inliers = emptySet
        for (each point that is not possible_inliers)
        {
            if satisfies possible_model i.e. error < t
                add the opint into also_inliers 
        }
        if (number of sample points in also_inliers > d) 
        {
            better_model = use all possible_inliers and also_inliers 
                          regenerate a better model
            this_err = errors of all possible_inliers and also_inliers
            if this_err < best_err
            {
                best_fit = better_model
                best_err = this_err
            }
        }
        iterations++
    }
    return best_fit
    """
    iterations = 0
    best_fit = None
    best_err = np.inf 
    best_inlier_idxs = None
    while iterations < k:
        possible_idxs, test_idxs = random_partition(n, data.shape[0]) 
        # get size(possible_idxs) rows of (Xi,Yi) data
        possible_inliers = data[possible_idxs, :]                               
        test_points = data[test_idxs]                 # rows of (Xi,Yi) data
        possible_model = model.fit(possible_inliers)  # fitting model
        # error: sum(squares)
        test_err = model.get_error(test_points, possible_model)  
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs,:]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(also_inliers) = %d' 
                   %(iterations, len(also_inliers)) )
        if len(also_inliers > d):
            # catenate the sample points
            better_data = np.concatenate( (possible_inliers, also_inliers) )      
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            this_err = np.mean(better_errs)          # new error: mean square                                      
            if this_err < best_err:
                best_fit = better_model
                best_err = this_err
                # update inliers, add new points
                best_inlier_idxs = np.concatenate( (possible_idxs, also_idxs) ) 
        iterations += 1
        if best_fit is None:
            raise ValueError("did not meet fitting acceptance criteria")
        if return_all:
            return best_fit,{'inliers':best_inlier_idxs}
        else:
            return best_fit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)           # get all indexes of n_data
    np.random.shuffle(all_idxs)            # disrupt the data
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquareModel:
    # Use least square to generate linear solution
	# Use the linear solution as the input of RANSAC   
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    def fit(self, data):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T              
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T            
        x, resids, rank, s = sl.lstsq(A, B) 
        return x    

    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T 
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T 
        BFit = sp.dot(A, model)                # BFit = model.k*A + model.b
        # sum squared error per row
        err_per_point = np.sum( (B - BFit) ** 2, axis = 1 )      
        return err_per_point

def test():
    # Generate ideal data
    n_samples = 500                             # number of sample points
    n_inputs = 1                                # number of inputs
    n_outputs = 1                               # number of outputs
    # randomly generate 500 data between 0-20: row vectors
    A_exact = 20 * np.random.random((n_samples, n_inputs))  
    # randomly generate a slope
    perfect_fit = 60 * np.random.normal( size = (n_inputs, n_outputs) )           
    B_exact = sp.dot(A_exact, perfect_fit)        # y = x * k

    # Add Gaussian noise
    A_noise = A_exact + np.random.normal( size = A_exact.shape ) 
    B_noise = B_exact + np.random.normal( size = B_exact.shape ) 

    if 1:
        # Add outliers
        nOutliers = 100
        all_idxs = np.arange( A_noise.shape[0] )  # get indexes 0-499
        np.random.shuffle(all_idxs)              # disrupt all_idxs
        outlier_idxs = all_idxs[:nOutliers]       # 100 random outliers of 0-500
        A_noise[outlier_idxs] = 20 * np.random.random( (nOutliers, n_inputs) )    
        B_noise[outlier_idxs] = 50 * np.random.normal( size = 
                                                     (nOutliers, n_outputs))
		
    # Setup model 
    all_data = np.hstack( (A_noise, B_noise) )                             
    input_columns = range(n_inputs)   
    output_columns = [n_inputs + i for i in range(n_outputs)]  
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug) 

    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], 
                                              all_data[:,output_columns])
    
    # Run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 
                                   300, debug = debug, return_all = True)

    if 1:
        import pylab

        sortIdxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sortIdxs] 

        if 1:
            pylab.plot( A_noise[:,0], B_noise[:,0], 'k.', label = 'data' ) 
            pylab.plot( A_noise[ransac_data['inliers'], 0], 
                B_noise[ransac_data['inliers'], 0], 'bx', label = "RANSAC data")
        else:
            pylab.plot( A_noise[non_outlier_idxs,0], 
                        B_noise[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noise[outlier_idxs,0], B_noise[outlier_idxs,0], 
                        'r.', label='outlier data' )

        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fitting' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='original system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fitting' )
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()
