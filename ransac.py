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
		bestFit - the optimal fitting solution
		        - return nil if not found
    
	Pseudocode:
	
    iterations = 0
    bestFit = nil                          # update later
    bestErr = something really large       # update later that bestErr = thisErr
    while iterations < k 
    {
        possibleInliers = randomly select n sample points
        possibleModel = n possibleInliers 
        alsoInliers = emptySet
        for (each point that is not possibleInliers)
        {
            if satisfies possibleModel i.e. error < t
                add the opint into alsoInliers 
        }
        if (number of sample points in alsoInliers > d) 
        {
            betterModel = use all possibleInliers and alsoInliers regenerate a better model
            thisErr = errors of all possibleInliers and alsoInliers
            if thisErr < bestErr
            {
                bestFit = betterModel
                bestErr = thisErr
            }
        }
        iterations++
    }
    return bestFit
    """
    iterations = 0
    bestFit = None
    bestErr = np.inf 
    best_inlier_idxs = None
    while iterations < k:
        possibleIndexs, testIdxs = random_partition(n, data.shape[0]) 
        possibleInliers = data[possibleIndexs, :]                               # get size(possibleIndexs) rows of (Xi,Yi) data
        testPoints = data[testIdxs]                                             # some rows of (Xi,Yi) data
        possibleModel = model.fit(possibleInliers)                              # fitting model
        testErr = model.get_error(testPoints, possibleModel)                    # error: sum(squares)
        alsoIdxs = testIdxs[testErr < t]
        alsoInliers = data[alsoIdxs,:]
        if debug:
            print ('testErr.min()',testErr.min())
            print ('testErr.max()',testErr.max())
            print ('numpy.mean(testErr)',numpy.mean(testErr))
            print ('iteration %d:len(alsoInliers) = %d' %(iterations, len(alsoInliers)) )
        if len(alsoInliers > d):
            betterData = np.concatenate( (possibleInliers, alsoInliers) )       # catenate the sample points
            betterModel = model.fit(betterData)
            better_errs = model.get_error(betterData, betterModel)
            thisErr = np.mean(better_errs)                                      # new error: mean square
            if thisErr < bestErr:
                bestFit = betterModel
                bestErr = thisErr
                best_inlier_idxs = np.concatenate( (possibleIndexs, alsoIdxs) ) # update inliers, add new points
        iterations += 1
        if bestFit is None:
            raise ValueError("did not meet fitting acceptance criteria")
        if return_all:
            return bestFit,{'inliers':best_inlier_idxs}
        else:
            return bestFit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    allIdxs = np.arange(n_data)                                                 # get all indexes of n_data
    np.random.shuffle(allIdxs)                                                  # disrupt the data
    idxs1 = allIdxs[:n]
    idxs2 = allIdxs[n:]
    return idxs1, idxs2

class LinearLeastSquareModel:
    # Use least square to generate linear solution
	# Use the linear solution as the input of RANSAC   
    def __init__(self, inputColumns, outputColumns, debug = False):
        self.inputColumns = inputColumns
        self.outputColumns = outputColumns
        self.debug = debug
    
    def fit(self, data):
        A = np.vstack( [data[:,i] for i in self.inputColumns] ).T              
        B = np.vstack( [data[:,i] for i in self.outputColumns] ).T            
        x, resids, rank, s = sl.lstsq(A, B) 
        return x    

    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.inputColumns] ).T 
        B = np.vstack( [data[:,i] for i in self.outputColumns] ).T 
        BFit = sp.dot(A, model)                                                 # BFit = model.k*A + model.b
        err_per_point = np.sum( (B - BFit) ** 2, axis = 1 )                     # sum squared error per row
        return err_per_point

def test():
    # Generate ideal data
    nSamples = 500                                                              # number of sample points
    nInputs = 1                                                                 # number of inputs
    nOutputs = 1                                                                # number of outputs
    AExact = 20 * np.random.random((nSamples, nInputs))                         # randomly generate 500 data between 0-20: row vectors
    perfectFit = 60 * np.random.normal( size = (nInputs, nOutputs) )            # randomly generate a slope
    BExact = sp.dot(AExact, perfectFit) # y = x * k

    # Add Gaussian noise
    ANoise = AExact + np.random.normal( size = AExact.shape ) 
    BNoise = BExact + np.random.normal( size = BExact.shape ) 

    if 1:
        # Add outliers
        nOutliers = 100
        allIdxs = np.arange( ANoise.shape[0] )                                  # get indexes 0-499
        np.random.shuffle(allIdxs)                                              # disrupt allIdxs
        outlierIdxs = allIdxs[:nOutliers]                                       # 100 random outliers of 0-500
        ANoise[outlierIdxs] = 20 * np.random.random( (nOutliers, nInputs) )    
        BNoise[outlierIdxs] = 50 * np.random.normal( size = (nOutliers, nOutputs))
		
    # Setup model 
    allData = np.hstack( (ANoise, BNoise) )                             
    inputColumns = range(nInputs)   
    outputColumns = [nInputs + i for i in range(nOutputs)]  
    debug = False
    model = LinearLeastSquareModel(inputColumns, outputColumns, debug = debug) 

    linearFit,resids,rank,s = sp.linalg.lstsq(allData[:,inputColumns], allData[:,outputColumns])
    
    # Run RANSAC algorithm
    ransacFit, ransacData = ransac(allData, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

    if 1:
        import pylab

        sortIdxs = np.argsort(AExact[:,0])
        A_col0_sorted = AExact[sortIdxs] 

        if 1:
            pylab.plot( ANoise[:,0], BNoise[:,0], 'k.', label = 'data' ) 
            pylab.plot( ANoise[ransacData['inliers'], 0], BNoise[ransacData['inliers'], 0], 'bx', label = "RANSAC data" )
        else:
            pylab.plot( ANoise[non_outlierIdxs,0], BNoise[non_outlierIdxs,0], 'k.', label='noisy data' )
            pylab.plot( ANoise[outlierIdxs,0], BNoise[outlierIdxs,0], 'r.', label='outlier data' )

        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransacFit)[:,0],
                    label='RANSAC fitting' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfectFit)[:,0],
                    label='original system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linearFit)[:,0],
                    label='linear fitting' )
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()