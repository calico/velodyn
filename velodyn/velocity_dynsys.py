"""
Dynamical systems simulations in RNA velocity space
"""
import numpy as np
from scipy import stats
import anndata
import tqdm
import typing
from typing import Collection
import warnings
# multiprocessing tools. pathos uses `dill` rather than `pickle`,
# which provides more robust serialization.
from pathos.multiprocessing import ProcessPool
from sklearn.neighbors import NearestNeighbors
# plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


class PhaseSimulation(object):
    """Perform phase point simulations in velocity fields.

    Attributes
    ----------
    adata : anndata.AnnData
        [Cells, Genes] object with precomputed attributes
        for RNA velocity in `.layers`.
        keys: {velocity, spliced, unspliced}.
    vadata : anndata.AnnData
        view of `.adata` used for velocity field estimation.
    pfield : np.ndarray
        [Cells, Features] positions of cells in the velocity field.
    vfield : np.ndarray
        [Cells, Features] velocities of cells in the velocity field.
    starting_points : np.ndarray
        [Cells, Features] starting points for phase points in the
        velocity field.
    v_model : Callable
        a model of RNA velocity that predicts velocity given a positional
        coordinate in the desired basis.
    trajectories : np.ndarray
        [PhasePoints, Time, Dimensions, (Position, V_mu, V_sig)]
        trajectories of phase points in the velocity field.
    boundary_fence : dict
        {"min", "max"} specifies fence conditions if the boundary constraint
        is set to obey a predefined fence. minimum and maximum values for
        each dimension are stored as lists.
    timesteps : int
        [T,] number of timesteps for phase point evolution.
    step_scale : float
        scaling factor for phase point steps in the chosen basis.
    noise_scale : float
        scaling factor for noise introduced during phase point evolution.
        defaults to a noiseless simulation.
    velocity_k : int
        number of nearest neighbors to consider when employing a 'knn'
        velocity model.
    vknn_method : str
        method by which to the kNN model computes velocity estimates for
        phase points.
        "deterministic" -- use the mean of kNN RNA velocity vectors.
        "stochastic" -- fit a multivar. Gaussian to kNN vectors and sample.
        "knn_random_sample" -- randomly sample an observed vector from kNN.
    Methods
    -------
    boundary_contraint(position, velocity)
        impose a boundary constraint by modifying the predicted position
        of an evolving phase point. defaults to an identity function.
    v_fxn : callable
        returns velocity as a function of position in the
        embedding space. takes a [D,] np.ndarray as input, returns
        a [D,] np.ndarray.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        **kwargs,
    ) -> None:
        """Perform phase point simulations in velocity fields.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes] object with precomputed attributes
            for RNA velocity in `.layers`.
            keys: {velocity, spliced, unspliced}.

        Returns
        -------
        None.
        """
        self.adata = adata
        if self.adata.raw is not None:
            print('`adata.raw` is not `None`.')
            print('This can cause indexing issues with some anndata versions.')
            print('Consider setting `adata.raw = None`\n.')

        # set the number of nearest neighbors to use when inferring
        # phase point velocities
        self.velocity_k = 100
        if 'velocity_k' in kwargs.keys():
            self.velocity_k = kwargs['velocity_k']

        # set an identity function as our initial boundary constraint
        # until we choose a different one
        self.boundary_constraint = self._identity_placeholder
        return

    def set_velocity_field(
        self,
        groupby: str = None,
        group: typing.Any = None,
        basis: str = 'counts',
    ) -> None:
        """Set a subset of cells to use when defining the
        velocity field.

        Parameters
        ----------
        groupby : str
            column in `.adata.obs` to use for group selection.
        group : Any
            value in `groupby` to use for selecting cells.
        basis : str
            basis for setting the velocity field. must be one
            of {'counts', 'pca', 'umap', 'tsne'}.
            if not 'counts', must have 'velocity_%s'%basis attribute.

        Returns
        -------
        None. Sets `.vadata`.

        Notes
        -----
        Generates a view of `.adata` with only the selected
        cells, `.vadata`.
        Sets the `.vfield` and `.pfield` attribute with selected
        cells in the desired basis.
        """
        # ensure that arguments are valid
        if groupby is not None and group is None:
            raise ValueError('Must supply a `group` for cell selection.')
        if group is not None and groupby is None:
            raise ValueError('Must supply a `groupby` for cell selection.')
        
        # check that the specified basis is supported
        bases = ['counts', 'pca', 'umap', 'tsne']
        if basis not in bases:
            raise ValueError('%s is not a valid basis.' % basis)

        # if no grouping variable is provided
        # create a single "dummy" group
        if groupby is not None and group is not None:
            bidx = self.adata.obs[groupby] == group
        else:
            bidx = np.ones(self.adata.shape[0]).astype(np.bool)

        # get the relevant cells from the grouping
        self.vadata = self.adata[bidx, :].copy()

        # set the velocity field and position field using
        # cell observations
        if basis == 'counts':
            self.vfield = self.vadata.layers['velocity']
            self.pfield = self.vadata.X
        else:
            self.vfield = self.vadata.obsm['velocity_%s' % basis]
            self.pfield = self.vadata.obsm['X_%s' % basis]

        # convert to dense if not
        # TODO: make downstream ops sparse compatible
        if type(self.vfield) != np.ndarray:
            self.vfield = self.vfield.toarray()
        if type(self.pfield) != np.ndarray:
            self.pfield = self.pfield.toarray()

        return

    def _set_starting_point_metadata(
        self,
        groupby: str = None,
        group: typing.Any = None,
    ) -> None:
        """Set starting points for phase point simulations based
        on sample annotations.

        Parameters
        ----------
        groupby : str
            column in `.adata.obs` to use for group selection.
        group : Any
            value in `groupby` to use for selecting cells.

        Returns
        -------
        None. Sets `.starting_points`.
        """
        # check that arguments are valid
        if groupby is None or group is None:
            raise ValueError('must supply both groupby and group.')

        # set starting points as the designated positions in the 
        # position field
        bidx = self.vadata.obs[groupby] == group
        print(f'Found {sum(bidx)} points matching starting criteria.')
        self.starting_points = self.pfield[bidx, :]
        return

    def _set_starting_point_embedding(
        self,
        basis: str = None,
        borders: tuple = None,
    ) -> None:
        """Set starting points for phase point simulations based
        on embedding locations.

        Parameters
        ----------
        basis : str
            embedding basis to use for selection.
            expects `'X_'+basis` in `.obsm.keys()`.
            e.g. 'pca', 'umap', 'tsne'.
        borders : tuple
            [N,] minimum and maximum values in each dimension of the
            embedding to use for starting point selection.
            e.g. ((-1, 1), (-3, 1)) for a 2D embedding.

        Returns
        -------
        None. sets `.starting_points`.
        """
        # check that the basis is present
        bindices = []
        if 'X_'+basis not in self.adata.obsm.keys():
            raise ValueError(
                'X_%s is not an embedding in `.adata.obsm`.' % basis)
        embed = self.vadata.obsm['X_'+basis]

        # get all cells within the borders specified along each dimension
        for i, min_max in enumerate(borders):
            bidx = np.logical_and(
                embed[:, i] > min_max[0],
                embed[:, i] < min_max[1],
            )
            bindices.append(bidx)

        # use cells that meet all border criteria as starting points
        bidx = np.logical_and.reduce(bindices)
        self.starting_points = self.pfield[bidx, :]
        return

    def _set_starting_point_expression(
        self,
        genes: Collection[str] = None,
        min_expr_levels: Collection[float] = None,
        use_raw: bool = True,
    ) -> None:
        """Set starting points for phase point simulations based
        on gene expression levels.

        Parameters
        ----------
        genes : Collection[str]
            [N,] gene names to use for starting point selection.
        min_expr_levels : Collection[float]
            [N,] minimum expression level for each gene.
        use_raw : bool
            use the `.adata.raw.X` attribute for gene expression levels
            instead of `.adata.X`.

        Returns
        -------
        None. sets `.starting_points`.
        """
        # check argument validity
        if genes is None or min_expr_levels is None:
            raise ValueError('must supply both genes and min_expr_levels')

        if len(genes) != len(min_expr_levels):
            ll = (len(genes), len(min_expr_levels))
            raise ValueError(
                '%d genes and %d min_expr_levels, must be equal.' % ll)

        # tolerate singleton arguments begrudgingly
        if type(genes) == str:
            warnings.warn(
                'casting `genes` to list in `_set_starint_point_expression`.'
            )
            genes = [genes]
        if type(min_expr_levels) == float:
            min_expr_levels = [min_expr_levels]
            warnings.warn(
                'casting `min_expr_level` to list `_set_starint_point_expression`.'
            )

        if use_raw:
            ad = self.vadata.raw
        else:
            ad = self.vadata
            
        # get cells that express the relevant genes at the minimum
        # levels specified
        bindices = []
        for i, g in enumerate(genes):
            expr = ad[:, g].X
            if type(expr) != np.ndarray:
                expr = expr.toarray()
            bidx = expr > min_expr_levels[i]
            bindices.append(bidx)

        # take only cells meeting all criteria as starting points
        bidx = np.logical_and.reduce(bindices)
        self.starting_points = self.pfield[bidx, :]
        return

    def set_starting_point(
        self,
        method: str,
        **kwargs,
    ) -> None:
        """Set starting points for phase point simulations.
        Uses metadata, embedding locations, or gene expression values.

        Parameters
        ----------
        method : str
            {'metadata', 'embedding', 'expression'}.
        **kwargs : dict
            passed to the relevant `._set_starting_point_{method}` function.

        Returns
        -------
        None. sets `.starting_points`.

        Notes
        -----
        Calls the relevant method for setting starting points based
        on the `method` argument and passes remaining keyword arguments.
        """
        # check argument validity
        acceptable_methods = ['metadata',  'embedding', 'expression']
        if method not in acceptable_methods:
            raise ValueError('%s is not an acceptable method.' % method)

        if not hasattr(self, 'pfield'):
            raise ValueError(
                'must set a `pfield` with `set_velocity_field` first.')

        f = getattr(self, '_set_starting_point_'+method)
        f(**kwargs)
        return

    def _identity_placeholder(
        self,
        x: typing.Any,
    ) -> typing.Any:
        """An identity function that returns an argument
        without modification. Useful as a placeholder."""
        return x

    def _boundary_constraint_fence(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Imposes a boundary constraint on phase point position
        `x` by forcing each dimension to sit within a pre-defined
        fence.

        Parameters
        ----------
        x : np.ndarray
            [D,] position of a phase point.

        Returns
        -------
        x_constrained : np.ndarray
            [D,] position of the phase point with dimensions clamped
            to a pre-defined fence.
        """
        # clip dimensions to fit within the boundary
        x_constrained = np.clip(
            x,
            self.boundary_fence['min'],
            self.boundary_fence['max'],
        )
        return x_constrained

    def _boundary_constraint_nn_dist(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Imposes a boundary constraint on phase point position
        `x` by forcing `x` to the nearest point that is less than
        a predefined distance from its nearest neighbors.

        Parameters
        ----------
        x : np.ndarray
            [D,] position of a phase point.

        Returns
        -------
        x_constrained : np.ndarray
            [D,] position of the phase point with dimensions clamped
            to a pre-defined fence.

        Notes
        -----
        Phase points are contrained to a maximum distance from their
        nearest neighbor. This distance can be adaptively determined
        by taking the median nearest neighbor distance from the data
        set and using some multiple of this distance as the boundary
        constraint.

        When a phase point passes beyond this distance, a distance
        vector is computed between the point and the neighbor, and
        the point location is shrunken along the vector to satisfy
        the boundary constraint.

        See Also
        --------
        `.set_boundaries`.
        """
        if len(x.shape) == 1:
            # pad to a [1, N] matrix for sklearn
            x = np.expand_dims(x, 0)
        # compute the distance to the nearest neighbor
        distances, indices = self.boundary_nn.kneighbors(x)
        if distances[0, 0] < self.max_nn_distance:
            x_constrained = x
        else:
            nn_point = self.pfield[indices[0, 0]:indices[0, 0]+1, :]
            d_vec = x - nn_point
            # how much larger is the difference vector than what we allow?
            scale_factor = self.max_nn_distance / distances[0, 0]
            # scale the difference vector and compute x_constrained
            # as this scaled vector moving away from the NN
            d_vec *= scale_factor
            x_constrained = nn_point + d_vec
        return x_constrained

    def set_boundaries(
        self,
        method: str = 'fence',
        borders: tuple = None,
        max_nn_distance: float = None,
        boundary_knn: int = 5,
    ) -> None:
        """Impose boundaries for phase point simulations.
        During evolution, phase points will not move beyond
        these boundaries. This can prevent numerical instability
        issues where a phase point travels "off the map".

        Parameters
        ----------
        method : str
            one of {'fence', 'nn'}.
            fence - restrict phase points to a "fence" of the basis described
            with minimum and maximum values for each dimension.
            nn - restrict phase points to a maximum distance away from their
            nearest neighbor. this maximum distance is determined either
            empirically or by taking the median nearest neighbor distance
            from the data set. when points travel beyond this distance, they
            are shrunken back toward the neighbor along the distance vector.
        borders : tuple
            ((min_i, max_i), ...) for each dimension of the basis.
            only used if `method` is "fence".
        max_nn_distance : float
            maximum distance a phase point may travel from the
            nearest neighbor. if `None`, set to the median nearest neighbor
            distance in the data set.
            only used if `method` is "nn".
        boundary_knn : int
            number of nearest neighbors to use for 'nn' boundary fencing.
            moves cells toward the centroid of this nearest neighbor group.

        Returns
        -------
        None. Sets `.boundary_constraint` attribute.

        See Also
        --------
        _boundary_constraint_fence
        _boundary_constraint_nn_distance
        """
        # check argument validity
        if method not in ('fence', 'nn'):
            raise NotImplementedError(
                '%s is not an implemented method.' % method)

        if method.lower() == 'fence':
            if borders is None:
                raise ValueError('must specify borders if method is fence.')
            # unpack border criteria into an attribute
            self.boundary_fence = {}
            self.boundary_fence['min'] = [x[0] for x in borders]
            self.boundary_fence['max'] = [x[1] for x in borders]
            # set the boundary contraint function to consider
            # the border fence during phase point updates
            self.boundary_constraint = self._boundary_constraint_fence
        elif method.lower() == 'nn':
            # the "nearest neighbor" to each point after fitting the NN
            # model is the point itself, so we fit k = 2 here and take
            # the "second" nearest neighbor for each point when predicting
            # on the points themselves. Note that since phase points aren't
            # in the training set, we subsequently use only the first neighbor.
            self.boundary_nn = NearestNeighbors(
                n_neighbors=2, metric='euclidean')
            self.boundary_nn.fit(self.pfield)
            if max_nn_distance is None:
                # Compute nearest neighbor distances in the data set
                if not hasattr(self, 'pfield'):
                    raise ValueError(
                        'must `set_velocity_field` before NN boundaries.')
                distances, indices = self.boundary_nn.kneighbors(self.pfield)
                median_distance = np.median(
                    distances[:, 1:self.boundary_knn+1])
                self.max_nn_distance = median_distance
            else:
                self.max_nn_distance = max_nn_distance
            self.boundary_constraint = self._boundary_constraint_nn_dist
        return

    def _velocity_knn(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Calculate the velocity of a given position based
        on the average velocity of the k-NN to that position.

        Parameters
        ----------
        x : np.ndarray
            [D,] position vector in embedding space.

        Returns
        -------
        nn_v : np.ndarray
            [D, (Mean, Std)] velocity vector in embedding space.

        See Also
        --------
        .k
        """
        # find nearest neighbors
        nn_dist, nn_idx = self.v_nn.kneighbors(
            x.reshape(1, -1),
            return_distance=True,
        )

        nn_idx = nn_idx.flatten()

        # calculate the velocity vector
        if self.vknn_method == 'deterministic':
            nn_v_mu = self.vfield[nn_idx, :].mean(0)
        elif self.vknn_method == 'stochastic':
            # fit a multivariate Gaussian to the observed
            # RNA velocity vectors of the nearest neighbors

            # compute weights for each neighboring cell
            weights = stats.norm.pdf(x=nn_dist, scale=self.mean_nn_distance)

            weights_mat = np.tile(
                weights.reshape(-1, 1),
                (1, self.vfield.shape[1]),
            )
            mu = np.sum(weights_mat*self.vfield[nn_idx, :], 0)/np.sum(weights)
            # get weighted covariance
            # \Sigma = \frac{1}{\sum_{i=1}^{N} w_i - 1}
            #   {\sum_{i=1}^N w_i \left(x_i - \mu^*\right)^T\left(x_i - \mu^*\right)}

            cov = np.cov(
                self.vfield[nn_idx, :],
                aweights=weights.flatten(),
                rowvar=False,
            )
            
            # init a multivariate normal with the weighted
            # mean and covariance
            norm = stats.multivariate_normal(
                mean=mu,
                cov=cov,
            )
            # sample from the fitted Gaussian
            nn_v_mu = norm.rvs()
        elif self.vknn_method == 'knn_random_sample':
            # randomly sample a velocity vector 
            # from one of the nearest neighbors
            ridx = int(np.random.choice(nn_idx, size=1))
            nn_v_mu = self.vfield[ridx, :]
        else:
            msg = f'{self.vknn_method} is not a valid method for ._velocity_knn'
            raise AttributeError(msg)

        nn_v_sd = self.vfield[nn_idx, :].std(0)
        nn_v = np.stack([nn_v_mu, nn_v_sd], -1)
        return nn_v

    def _evolve(
        self,
        x0_idx: int,
    ) -> np.ndarray:
        """
        Place a phase point at `x0` and evolve for `t` timesteps.

        Parameters
        ----------
        x0_idx : int
            index for starting point `self.starting_points`.

        Returns
        -------
        trajectory : np.ndarray
            [T, D, (Position, V_mu, V_sig)] trajectory of the
            phase point.
        """
        x0 = self.starting_points[x0_idx, :]
        if type(x0) != np.ndarray:
            x0 = x0.toarray()
        x0 = x0.flatten()
        # [T, Dims, (Position, Velocity)]
        trajectory = np.zeros(
            (self.timesteps, x0.shape[0], 3), dtype=np.float32)
        
        # for each timestep, update the position of the phase point
        # based on the velocity of nearest neighbors and obey any
        # boundary constraints
        x = x0
        for t in range(self.timesteps):
            trajectory[t, :, 0] = x  # match x position to dv/dx
            v = self.v_fxn(x=x.reshape(-1),)
            # add white noise if desired to better emulate a stochastic process
            noise = v[:, 1] * np.random.randn(v.shape[0]) * self.noise_scale
            x_new = x + (v[:, 0] + noise)*self.step_scale
            # constrain to a set of pre-defined boundaries
            # defaults to an identity if not set explicitly
            x_new = self.boundary_constraint(x_new)
            trajectory[t, :, 1] = v[:, 0]
            trajectory[t, :, 2] = v[:, 1]
            x = x_new
        return trajectory

    def _evolve2disk(self, **kwargs) -> str:
        """Performs phase point evolution, but saves results to disk rather
        than returning the array."""
        raise NotImplementedError('evolve2disk is not yet implemented.')

    def __getstate__(self) -> dict:
        """Redefine __getstate__ to allow serialization of class methods.
        `anndata.AnnData` doesnt support serialization.
        """
        self_dict = self.__dict__.copy()
        # we remove large objects from `__getstate__` to allow
        # pickling for `multiprocessing.Pool` workers without
        # high memory overhead
        del self_dict['adata']
        del self_dict['vadata']
        return self_dict

    def simulate_phase_points(
        self,
        n_points: int = 1000,
        n_timesteps: int = 1000,
        velocity_method: str = 'knn',
        velocity_method_attrs: dict = {
            'vknn_method': 'deterministic',
        },
        step_scale: float = 1.,
        noise_scale: float = 0.,
        multiprocess: bool = False,
    ) -> np.ndarray:
        """Simulate phase points moving through the velocity field.

        Parameters
        ----------
        n_points : int
            number of points to simulate.
        n_timesteps : int
            number of timesteps for evolution.
        velocity_method : str
            method for estimating velocity during phase point evolution.
            one of {'knn', 'v_model'}.
            if 'v_model', must set the `.v_model` attribute with a Callable
            that takes in a position and outputs a velocity. useful if you
            want to train a model to map positions to velocities.
        velocity_method_attrs: dict
            attributes for use in a particular velocity method.
            keys are attribute names added to `self` with corresponding
            values.
        step_scale : float
            scaling factor for steps in the embedding space.
        noise_scale : float
            scaling factor for noise introduced during simulation.
            defaults to a noiseless simulation.     
        multiprocess : bool
            use multiprocessing.

        Returns
        -------
        trajectories : np.ndarray
            [PhasePoints, Time, Dimensions, (Position, V_mu, V_sig)]
            trajectories of phase points in the velocity field.
        also sets `.trajectories` attribute.

        Notes
        -----
        TODO: multithread these operations
        """
        # check argument validity
        if velocity_method not in ['knn', 'v_model']:
            raise ValueError(
                '%s is not a valid velocity method.' % velocity_method)

        if not hasattr(self, 'vfield') or not hasattr(self, 'pfield'):
            raise ValueError(
                'must first set velocity field with `set_velocity_field`.')

        if not hasattr(self, 'starting_points'):
            raise ValueError(
                'must first set starting points with `set_starting_points`.')

        if velocity_method == 'knn':
            self.v_fxn = self._velocity_knn
            if 'vknn_method' not in velocity_method_attrs:
                msg = 'velocity_method knn requires a "vknn_method" attribute.'
                raise ValueError(msg)
            # fit a nearest neighbors model to the data
            self.v_nn = NearestNeighbors(n_neighbors=self.velocity_k)
            self.v_nn.fit(self.pfield)

            # get the mean distance between nearest neighbors
            d, _ = self.v_nn.kneighbors(self.pfield)
            self.mean_nn_distance = d[:, 1].mean()

        elif velocity_method == 'v_model':
            if not hasattr(self, 'v_model'):
                raise ValueError('must specify a neural network model first.')
            self.v_fxn = self.v_model
        else:
            msg = f'{velocity_method} is not a valid velocity method.'
            raise ValueError(msg)

        if velocity_method_attrs is not None:
            # add the velocity method attrs to self
            for k in velocity_method_attrs.keys():
                setattr(self, k, velocity_method_attrs[k])

        self.timesteps   = n_timesteps
        self.step_scale  = step_scale
        self.noise_scale = noise_scale

        if multiprocess:
            # get a set of starting locations
            ridx = np.random.choice(np.arange(self.starting_points.shape[0]),
                                    size=n_points,
                                    replace=True)
            # open a process pool
            p = ProcessPool()
            # distribute tasks to workers
            res = p.map(self._evolve, ridx.tolist())
            p.close()
            # aggregate trajectory results
            trajectories = np.stack(res, 0)
        else:
            trajectories = np.zeros(
                (
                    n_points,
                    n_timesteps,
                    self.pfield.shape[1],
                    3,
                ),
                dtype=np.float32,
            )
            for i in tqdm.tqdm(
                range(n_points),
                desc='simulating trajectories'
            ):

                # select a random starting point
                ridx = np.random.choice(
                    np.arange(self.starting_points.shape[0]),
                    size=1,
                    replace=False,
                )

                # simulate the trajectory!
                phase_traj = self._evolve(x0_idx=ridx,)
                trajectories[i, :, :, :] = phase_traj

        self.trajectories = trajectories
        return trajectories


##########################################
# plotting methods
##########################################

def plot_phase_simulations(
    adata: anndata.AnnData,
    trajectories: np.ndarray,
    basis: str = 'pca',
    figsize: tuple = (6, 4),
    point_color='lightgray',
    trajectory_cmap='Purples',
    n_colors: int = 40,
    **kwargs,
) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
    """Plot phase simulation trajectories.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment object.
    trajectories : np.ndarray
        [PhasePoints, Time, Dimensions, (Position, V_mu, V_sig)]
        trajectories of phase points in the velocity field.
    basis : str
        coordinate basis in `adata.obsm` to use.
        retrieves `adata.obsm[f'X_{basis}']`.
    figsize : tuple
        (W, H) for matplotlib figure.
    point_color : str
        color to use for observed cell coordinate points.
    trajectory_cmap : str
        colormap to use for plotting trajectories.
        single color maps (e.g. "Purples", "Blues") work well.
    n_colors : int
        number of steps in the color gradient and number of unique
        points to plot for each trajectory.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    E = adata.obsm[f'X_{basis}']

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(
        E[:, 0],
        E[:, 1],
        color=point_color,
        alpha=0.5,
    )

    n_steps = trajectories.shape[1]

    gradient = sns.color_palette(trajectory_cmap, n_colors)
    for i, t in enumerate(
        np.arange(0, n_steps, n_steps//n_colors)[:-1][:n_colors]
    ):
        T = trajectories[:, t, :, 0]
        ax.scatter(
            T[:, 0],
            T[:, 1],
            color=gradient[i],
            **kwargs,
        )
    ax.set_xlabel(f'{basis} 1')
    ax.set_ylabel(f'{basis} 2')
    ax.set_title(f'Phase Points - {basis} Basis')
    return fig, ax
