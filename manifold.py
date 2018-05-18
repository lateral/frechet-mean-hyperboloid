import numpy as np

def coordinate_vector(i, n):
    """
    Return the length n 1D np.ndarray that is one hot in position 1.
    """
    vec = np.zeros(n, dtype=np.float64)
    vec[i] = 1
    return vec

def retraction(hyperboloid_pt, hyperboloid_tangent):
    """
    Perform an additive-Nickel&Kiela-style update after descending to the
    Poincaré ball.  Afterwards, come back to the hyperboloid.
    """
    poincare_pt = to_poincare_ball_point(hyperboloid_pt)
    poincare_tangent = to_poincare_ball_tangent(hyperboloid_pt, hyperboloid_tangent)
    poincare_pt += poincare_tangent
    norm = np.sqrt(poincare_pt.dot(poincare_pt))
    if norm >= 1:
        poincare_pt *= ((1 - 1e-5) / norm)
    return to_hyperboloid_point(poincare_pt)

def distance(u, v):
    """
    `u` and `v` are vectors on the forward hyperboloid in Minkowski space.
    """
    return np.arccosh(-minkowski_dot(u, v))

def minkowski_dot(u, v):
    """
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:rank].dot(v[:rank])
    return euc_dp - u[rank] * v[rank]

def logarithm(base, other):
    """
    Return the logarithm of `other` in the tangent space of `base`.
    """
    mdp = minkowski_dot(base, other)
    dist = np.arccosh(-mdp)
    proj = other + mdp * base
    norm = np.sqrt(minkowski_dot(proj, proj)) 
    if norm > 1e-10:
        proj *= dist / norm
    return proj

def exponential(base, tangent):
    """
    Compute the exponential of `tangent` from the point `base`.
    """
    tangent = tangent.copy()
    norm = np.sqrt(max(minkowski_dot(tangent, tangent), 0))
    if norm == 0:
        return base
    tangent /= norm
    return np.cosh(norm) * base + np.sinh(norm) * tangent

def geodesic_parallel_transport(base, direction, tangent):
    """
    Parallel transport `tangent`, a tangent vector at point `base`, along the
    geodesic in the direction `direction` (another tangent vector at point
    `base`, not necessarily unit length)
    """
    norm_direction = np.sqrt(minkowski_dot(direction, direction))
    unit_direction = direction / norm_direction
    parallel_component = minkowski_dot(tangent, unit_direction)
    unit_direction_transported = np.sinh(norm_direction) * base + np.cosh(norm_direction) * unit_direction
    return parallel_component * unit_direction_transported + tangent - parallel_component * unit_direction 

def frechet_gradient(theta, points, weights=None):
    """
    Return the gradient of the weighted Frechet mean of the provided points at the hyperboloid point theta.
    This is tangent to theta in on the hyperboloid.
    Arguments are numpy arrays.  `points` is 2d, the others are 1d. They satisfy:
    len(weights) == len(points) and points.shape[1] == theta.shape[0].
    If weights is None, use uniform weighting.
    """
    if weights is None:
        weights = np.ones_like(points[:,0]) / points.shape[0]
    weights /= weights.sum()
    last = theta.shape[0] - 1
    mdps = points[:,:last].dot(theta[:last]) - points[:,last] * theta[last]
    max_mdp = -(1 + 1e-10)
    mdps[mdps > max_mdp] = max_mdp
    dists = np.arccosh(-mdps)
    scales = -dists * weights / np.sqrt(mdps ** 2 - 1)
    minkowski_tangent = (points * scales[:,np.newaxis]).sum(axis=0)
    return project_onto_tangent_space(theta, minkowski_tangent)

def project_onto_tangent_space(hyperboloid_point, minkowski_tangent):
    return minkowski_tangent + minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point

def to_poincare_ball_tangent(hyperboloid_pt, hyperboloid_tangent):
    N = len(hyperboloid_pt) - 1
    denom = hyperboloid_pt[N] + 1
    return (hyperboloid_tangent[:N] - ((hyperboloid_tangent[N] / denom) * hyperboloid_pt[:N])) / denom

def to_poincare_ball_point(hyperboloid_pt):
    """
    Project the point of the hyperboloid onto the Poincaré ball.
    Post: len(result) == len(hyperboloid_pt) - 1
    """
    N = len(hyperboloid_pt) - 1
    return hyperboloid_pt[:N] / (hyperboloid_pt[N] + 1)

def to_hyperboloid_points(poincare_pts):
    """
    Post: result.shape[1] == poincare_pts.shape[1] + 1
    """
    norm_sqd = (poincare_pts ** 2).sum(axis=1)
    N = poincare_pts.shape[1]
    result = np.zeros((poincare_pts.shape[0], N + 1), dtype=np.float64)
    result[:,:N] = (2. / (1 - norm_sqd))[:,np.newaxis] * poincare_pts
    result[:,N] = (1 + norm_sqd) / (1 - norm_sqd)
    return result

def to_hyperboloid_point(poincare_pt):
    """
    Post: len(result) == len(poincare_pt) + 1
    """
    return to_hyperboloid_points(poincare_pt[np.newaxis,:])[0,:]

def basepoint(dimension):
    """
    Return the basepoint of the hyperboloid with specified local dimension.
    Post: len(result) == dimension + 1
    """
    return np.eye(dimension + 1)[dimension,:]

def basepoint_tangent(i, dimension):
    """
    Return the ith basis vector of the tangent space of the basepoint of the
    hyperboloid with specified local dimension.
    Post: len(result) == dimension + 1
    """
    return np.eye(dimension + 1)[i,:]

def hyperboloid_circle(centrept, radius, number_points):
    """
    Return a list points on the 2-dimensional hyperboloid tracing out a circle
    with the given centre and radius.
    """
    assert len(centrept) == 3
    basept = coordinate_vector(2, 3)
    _log = logarithm(basept, centrept)
    distance = np.sqrt(minkowski_dot(_log, _log))
    if distance > 1e-10:
        tangents = [geodesic_parallel_transport(basept, _log, coordinate_vector(i, 3)) for i in range(2)]
    else:
        tangents = [coordinate_vector(i, 3) for i in range(2)]
    pts = []
    for angle in np.linspace(0, 2*np.pi, number_points):
        tangent = np.cos(angle) * tangents[0] + np.sin(angle) * tangents[1]
        pt = exponential(centrept, radius * tangent)
        pts.append(pt)
    return pts
