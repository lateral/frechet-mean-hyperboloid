import matplotlib.pyplot as plt
import numpy as np
from manifold import *

def plot_disc(size=8):
    """
    Plot a disc, returning the Figure and Axes.
    """
    fig = plt.figure(figsize=(size, size))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    circle = plt.Circle((0,0), 1., color='black', fill=False)
    ax.add_artist(circle)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return fig, ax

def plot_geodesic(ax, pt0, pt1, **kwargs):
    """
    Plots the disc on the geodesic.
    pt0, pt1 are hyperboloid points.
    """
    tangent = logarithm(pt0, pt1)
    ppts = []
    for step in np.linspace(0, 1, 100):
        ppts.append(to_poincare_ball_point(exponential(pt0, step * tangent)))
    ppts = np.array(ppts)
    ax.plot(ppts[:,0], ppts[:,1], **kwargs)
    
def scatterplot_on_disc(ax, hyperboloid_points, **kwargs):
    if len(hyperboloid_points.shape) == 1:
        hyperboloid_points = hyperboloid_points[np.newaxis,:]
    assert len(hyperboloid_points.shape) == 2
    ppoints = np.array([to_poincare_ball_point(pt) for pt in hyperboloid_points])
    ax.scatter(ppoints[:,0], ppoints[:,1], **kwargs)

def plot_on_disc(ax, hyperboloid_points, **kwargs):
    ppoints = np.array([to_poincare_ball_point(pt) for pt in hyperboloid_points])
    ax.plot(ppoints[:,0], ppoints[:,1], **kwargs)
