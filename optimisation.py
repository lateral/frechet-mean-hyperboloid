import matplotlib.pyplot as plt
import numpy as np
from manifold import *
from plotting import *

def distance_exceeds(point1, point2, threshold):
    """
    Return whether the hyperbolic distance between the two hyperboloid points
    provided exceeds `threshold` 
    """
    return -1 * minkowski_dot(point1, point2) > np.cosh(threshold)


MAX_STEPS = 500

class Optimisation:
    
    def __init__(self, points, init, target, update_fn, lr, tolerance):
        """
        Optimises for the Fréchet mean of `points` (which has been precomputed
        and is provided as the point `target`), starting from the point `init`,
        using the provided update function `update_fn` (either `exponential` or
        `retraction`) and learning rate `lr` > 0.  Stop the optimisation once
        arrived with a distance `tolerance` of `target`.
        All points are on the hyperboloid.
        `points` is a 2d numpy array, each row of which is a point.
        """
        self.points = points
        self.init = init
        self.target = target
        self.update_fn = update_fn
        self.lr = lr
        self.tolerance = tolerance
        self.scaled_gradients = []
        self.thetas = []
        self.arrived = False
        self._optimise()
        
    def hash_of_initial_conditions(self):
        """
        Return a hash that captures all of the initial parameters, with the
        exception of the update method.
        """
        return hash(str(self.points.tostring()) + str(self.init.tostring())\
                    + str(self.target.tostring()) + str(self.lr) + str(self.tolerance))

    def _optimise(self):
        """
        Calculate the intermediate steps in our optimisation, storing them in
        the list `self.thetas`, and the lr * gradients that led to them in
        `self.scaled_gradients`.  Perform as many steps as required (up to
        MAX_STEPS) to arrive with distance `tolerance` of `target` beginning
        from `init`, using `-lr` times the Frechet gradient w.r.t. `points` to
        update.
        Post:
        self.arrived == whether the optimisation arrived in in the
        neighbourhood of the solution.
        len(self.thetas) == len(self.scaled_gradients) + 1
        """
        self.thetas = [self.init]
        while(not self.arrived):
            if len(self.thetas) - 1 == MAX_STEPS:
                return
            gradient = frechet_gradient(self.thetas[-1], self.points)
            self.scaled_gradients.append(-1 * self.lr * gradient)
            self.thetas.append(self.update_fn(self.thetas[-1],
                                              self.scaled_gradients[-1]))
            self.arrived = not distance_exceeds(self.thetas[-1],
                                                self.target, self.tolerance)

    def get_number_updates(self):
        return len(self.scaled_gradients)
    
    def __str__(self):
        format_str = 'lr=%.2f; tolerance=%f; number of steps=%i; arrived=%s; update_fn=%s' 
        return (format_str % (self.lr, self.tolerance, self.get_number_updates(),
                              str(self.arrived), self.update_fn.__name__))
            
    def plot(self):
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        fig, ax = plot_disc()
        scatterplot_on_disc(ax, self.points, c='b', label='points')
        scatterplot_on_disc(ax, self.target, c='b', marker='^', label='Fréchet mean')
        scatterplot_on_disc(ax, self.init, c='g', label='initial parameters')
        scatterplot_on_disc(ax, np.array(self.thetas[1:]), c='g', label='updates', marker='x', alpha=0.5)
        for theta, scaled_grad_at_theta in zip(self.thetas[:-1], self.scaled_gradients):
            intermediate_pts = []
            for t in np.linspace(0, 1, 100):
                intermediate_pts.append(self.update_fn(theta, t * scaled_grad_at_theta))
            plot_on_disc(ax, intermediate_pts, c='g', linestyle='--', alpha=0.5)
        ax.legend(fontsize=14, bbox_to_anchor=(1.5, 1))
        ax.set_title(str(self))
        return fig, ax
