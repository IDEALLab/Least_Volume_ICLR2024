import torch

class LeastVolumeBinarySearch:
    """
    recursively find the optimal Lagrangian multiplier 
    by solving the concave dual problem with binary search.
    """
    def __init__(self, vol_func, rec_loss, eps, infL_optimizer, threshold=0.05, init_lambda=1e-2):
        self.vol_func = vol_func
        self.rec_loss = rec_loss
        self.eps = eps
        self.threshold = threshold
        self.infL_optimizer = infL_optimizer

        assert init_lambda > 0, 'λ should be positive!'
        self.lamb = init_lambda if init_lambda > 0 else 1 
        self._n = 0
        self._lamb_min = self._lamb_max = None
    
    def optimize(self, max_iter=500, verbose=False):
        """
        optimize dual problem, return is_converged, 
        lambda, volume and reconstruction loss.
        """
        #### update lambda ####
        self.update_lamb()

        #### derive inf L ####
        L = lambda: self.vol_func() - self.lamb * (self.eps - self.rec_loss())
        self.infL_optimizer(L) # should have a convergence criterion
        
        #### report ####
        if verbose: self.report()
        
        #### determine convergence ####
        if self.residual >= 0:
            self._lamb_max = self.lamb
            if torch.abs(self.residual)/self.eps <= self.threshold: # end recursion if is converged
                return True, self.lamb, self.vol_func(), self.rec_loss()
        elif self.residual < 0:
            self._lamb_min = self.lamb

        #### continue, or end recursion if n > max_iter ####
        self._n += 1
        if self._n >= max_iter: 
            return False, self.lamb, self.vol_func, self.rec_loss  
        else:
            return self.optimize(max_iter, verbose)
    
    @property
    def residual(self):
        """constraint is satisfied when it's >= 0"""
        return self.eps - self.rec_loss()
    
    def update_lamb(self):
        if self._lamb_min and self._lamb_max:
            self.lamb = (self._lamb_max + self._lamb_min) / 2
        elif self._lamb_min and not self._lamb_max:
            self.lamb = 2 * self._lamb_min
        elif self._lamb_max and not self._lamb_min:
            self.lamb = 0.5 * self._lamb_max
        else:
            pass
    
    def report(self):
        print('Iter #{}: λ = {}, obj = {}, residual = {}'.format(
            self._n, self.lamb, self.vol_func(), self.residual))