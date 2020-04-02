"""Compute divergence maps from RNA velocity fields"""
import numpy as np
import anndata

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# modified from
# https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_grid.py
def compute_velocity_on_grid(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    density: float = None,
    smooth: float = None,
    n_neighbors: int = None,
    min_mass: float = None,
    n_grid_points: int = 50,
    adjust_for_stream: bool = False,
    grid_min_max: tuple = None,
) -> (np.ndarray, np.ndarray):
    """
    Compute a grid of velocity vectors in gene expression space
    where each vector in the grid is a Gaussian weighted average of
    neighboring observed cell vectors.

    Parameters
    ----------
    X_emb : np.ndarray
        [Cells, (embedding0, embedding1)] cell coordinates in the
        embedding.
    V_emb : np.ndarray
        [Cells, (embedding0, embedding1)] cell velocities in the
        embedding.
    density : float
        [0, 1.] proportion of n_grid_points to use.
    smooth : float
        smoothing parameter for the Gaussian kernel.
    n_neighbors : int
        number of neighbors to consider.
    min_mass : float
        minimum probability mass to return a value for a grid cell.
    n_grid_points : int
        number of grid points along each dimension.
    adjust_for_stream : bool
        adjust grid velocities to be compatible with stream plots.
    grid_min_max : tuple
        ((min, max), (min, max)) values for coarse-graining grid
        coordinates. set manually to ensure coarse-grained coordinates
        are consistent across samples passed to `X_emb`.

    Returns
    -------
    X_grid : np.ndarray
        [n_grid_points, n_grid_points] locations of each vector
        in embedding space.
    V_grid : np.ndarray
        [n_grid_points, n_grid_points] RNA velocity vectors in
        the local neighborhood at a series of grid points.
    """
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = .5 if smooth is None else smooth

    # Generates a linearly spaced grid from the minimum to maximum
    # embedding coordinate along each dimension
    # the number of grid locations is specific with `n_grid_points`
    grs = []
    for dim_i in range(n_dim):
        if grid_min_max is None:
            m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
            m = m - .01 * np.abs(M - m)
            M = M + .01 * np.abs(M - m)
        else:
            m, M = grid_min_max[dim_i]
            pass
        gr = np.linspace(m, M, n_grid_points * density)
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    # find nearest neighbors to each grid point using `n_neighbors`
    # determine their relative distances
    if n_neighbors is None:
        n_neighbors = int(n_obs/50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    # [GridPoints, Dims] floats, array indices for nearest neighbors
    dists, neighs = nn.kneighbors(X_grid)

    # weight the contribution of each point with a Gaussian kernel
    # centered on the point of interest

    # here, `smooth` is a scaling factor that determines the sigma
    # of the Gaussian, which is the product of the total range of a dimension
    # and the scaling parameter
    # defaults to a sigma == 0.5*DimensionRange
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth

    # here, we evaluate a weight for each point as the PDF of a Gaussian with
    # the specified scale centered at the point, since we feed in distances
    # rather than coordinates
    weight = normal.pdf(x=dists, scale=scale)  # weights is [GridPoints, Dims]

    # p_mass stores how much probability mass is near a point
    # if all neighbors are very far away, this will be small
    p_mass = weight.sum(1)  # p_mass is [GridPoints,]

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / \
        np.maximum(1, p_mass)[:, None]

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        V_grid[0][mass.reshape(V_grid[0].shape) < 1e-5] = np.nan
    else:
        if min_mass is None:
            min_mass = np.clip(np.percentile(p_mass, 95) / 100, 1e-2, 1)
        # zero out vectors with little support
        V_grid[p_mass < min_mass] = 0.

    return X_grid, V_grid


def divergence(f):
    """
    Computes the divergence of the vector field.

    Parameters
    ----------
    f : list of ndarrays
        [D,] each array contains values for one dimension of
        the vector field.

    Returns
    -------
    D : np.ndarray
        divergence values in the same shape a items in `f`
        
    Notes
    -----
    The divergence of a vector field :math:`V(x, y)` is given by the sum of
    partial derivatives of the d-component with respect to d, where d is either
    x or y.
    
    .. math::
    
        \nabla V = \sum_{d \in \{x, y\}} \partial V_d(x, y) / \partial d
        
        \nabla V = \partial V_x(x, y)/\partial x + \partial V_y(x, y)/\partial y
    """
    num_dims = len(f)
    # for each dimension of the vector field `i`, compute the gradient with
    # respect to that dimension and add the results
    D = np.ufunc.reduce(
        np.add,
        [np.gradient(f[num_dims - i - 1], axis=i) for i in range(num_dims)]
    )
    return D


def compute_div(
    adata: anndata.AnnData,
    use_rep: str = 'pca',
    n_grid_points: int = 30,
    **kwargs,
) -> np.ndarray:
    """
    Compute divergence in gene expression space for a single
    cell experiment.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] single cell experiment containing velocity
        vectors for each cell.
    use_rep : str
        representation to use for divergence field calculation.
        `adata.obsm[f'X_{use_rep}']` and `adata.obsm[f'velocity_{use_rep}']`
        must be present.
    n_grid_points : int
        number of grid points along each dimension.
    **kwargs passed to `compute_velocity_on_grid`.

    Returns
    -------
    D : np.ndarray
        [n_grid_points, n_grid_points] divergence values.

    See Also
    --------
    compute_velocity_on_grid
    divergence
    """
    # compute a grid of positions and their Gaussian
    # weighted velocities across the embedding space
    X_grid, V_grid = compute_velocity_on_grid(
        adata.obsm[f'X_{use_rep}'][:, :2],
        adata.obsm[f'velocity_{use_rep}'][:, :2],
        n_grid_points=n_grid_points,
        **kwargs,
    )
    # reshape the grid points into an [X, Y, 2] matrix
    V_spatial = V_grid.reshape(
        n_grid_points, 
        n_grid_points, 
        2,
    )
    # compute the divergence
    D_spatial = divergence([V_spatial[:, :, i]
                            for i in range(V_spatial.shape[2])])
    return D_spatial


def plot_div(
    D_spatial,
    pal='PRGn',
    center: float = 0.,
    cbar_label='Divergence',
    xticklabels: bool = False,
    yticklabels: bool = False,
    figsize: tuple = (6, 4),
    **kwargs,
) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
    """Plot a heatmap of the divergence values in an RNA velocity field.

    Parameters
    ----------
    D_spatial : np.ndarray
        [n_grid_points, n_grid_points] divergence values.
    pal : Union[str, matplotlib.colors.Colormap]
        color map for divergence colors. can be a matplotlib
        named colormap.
    center : float
        value for centering a divergent colormap.
    cbar_label : str
        label for the colorbar.
    xticklabels : bool
        use x-axis tick labels.
    yticklabels : bool
        use y-axis tick labels.
    figsize : tuple
        (W, H) of the matplotlib figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        D_spatial,
        cmap=pal,
        ax=ax,
        center=center,
        cbar_kws={'label': cbar_label},
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        **kwargs,
    )
    ax.invert_yaxis()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    return fig, ax
