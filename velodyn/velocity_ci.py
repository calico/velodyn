"""Generate confidence intervals for RNA velocity models by bootstrapping
across reads.

Our bootstrapping procedure is as follows:

1. Given a spliced count matrix ([Cells, Genes]) S and an unspliced matrix U,
create a total counts matrix X = S + U.
2.1 For each cell X_i \in X, fit a multinomial distribution. Sample D (depth) reads
from each multinomial to create a sampled count distribution across genes \hat X_i.
2.2 For each gene g in \hat X_i, fit a binomial distribution Binom(n=\hat X_ig, p=\frac{S_ig}{X_ig})
which represents the distribution of spliced vs. unspliced counts.
2.3 Sample an estimate of the spliced counts for X_ig, \hat S_ig ~ Binom(n=X_ig, p=S_ig/X_ig).
Compute the conjugate unspliced read count \hat U_ig = \hat X_ig - \hat S_ig.
3. Given the complete bootstrapped samples \hat S, \hat U, estimate a bootstrapped
velocity vector for consideration.

Bootstrap samples of cell counts therefore have the same number of counts as the original
cell, preventing any issues due to differing library depths:

    \sum_i \sum_j X_{ij} \equiv \sum_i \sum_j \hat X_{ij}

"""
import numpy as np
import anndata
import scvelo as scv
import time
import os.path as osp
import argparse
import multiprocessing


class VelocityCI(object):
    """Compute confidence intervals for RNA velocity vectors

    Attributes
    ----------
    adata : anndata.AnnData
        [Cells, Genes] experiment with spliced and unspliced read
        matrices in `.layers` as "spliced", "unspliced", "ambiguous".
        `.X` should contain raw count values, rather than transformed
        counts.
    S : np.ndarray
        [Cells, Genes] spliced read counts.
    U : np.ndarray
        [Cells, Genes] unspliced read counts.
    A : np.ndarray
        [Cells, Genes] ambiguous read counts.

    Methods
    -------
    _sample_abundance_profile(x)
        sample a total read count vector from a multinomial fit
        to the observed count vector `x`.
    _sample_spliced_unspliced(s, u, a, x_hat)
        sample spliced, unspliced, and ambiguous read counts from
        a multinomial given a sample of total read counts `x_hat`
        and observed `s`pliced, `u`nspliced, `a`mbigious counts.
    _sample_matrices()
        samples a matrix of spliced, unspliced and ambiguous read
        counts for all cells and genes in `.adata`.
    _fit_velocity(SUA_hat,)
        fits a velocity model to sampled spliced, unspliced counts
        in an output from `_sample_matrices()`
    bootstrap_velocity(n_iter, embed)
        generate bootstrap samples of RNA velocity estimates using
        `_sample_matrices` and `_fit_velocity` sequentially.

    Notes
    -----
    Parallelization requires use of shared ctypes to avoid copying our
    large data arrays for each child process. See `_sample_matrices` for
    a discussion of the relevant considerations and solutions.
    Due to this issue, we have modified `__getstate__` such that pickling
    this object will not preserve all of the relevant data.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """Compute confidence intervals for RNA velocity vectors

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes] experiment with spliced and unspliced read
            matrices in `.layers` as "spliced", "unspliced", "ambiguous".
            `.X` should contain raw count values, rather than transformed
            counts.

        Returns
        -------
        None.
        """
        # check that all necessary layers are present
        if 'spliced' not in adata.layers.keys():
            msg = 'spliced matrix must be available in `adata.layers`.'
            raise ValueError(msg)
        if 'unspliced' not in adata.layers.keys():
            msg = 'unspliced matrix must be available in `adata.layers`.'
            raise ValueError(msg)
        if 'ambiguous' not in adata.layers.keys():
            msg = 'ambiguous matrix must be available in `adata.layers`.'
            raise ValueError(msg)

        # copy relevant layers in memory to avoid altering the original
        # input
        self.adata = adata
        self.S = adata.layers['spliced'].copy()
        self.U = adata.layers['unspliced'].copy()
        self.A = adata.layers['ambiguous'].copy()

        # convert arrays to dense format if they are sparse
        if type(self.S) != np.ndarray:
            try:
                self.S = self.S.toarray()
            except ValueError:
                msg = 'self.S was not np.ndarray, failed .toarray()'
                print(msg)

        if type(self.U) != np.ndarray:
            try:
                self.U = self.U.toarray()
            except ValueError:
                msg = 'self.U was not np.ndarray, failed .toarray()'
                print(msg)

        if type(self.A) != np.ndarray:
            try:
                self.A = self.A.toarray()
            except ValueError:
                msg = 'self.A was not np.ndarray, failed .toarray()'
                print(msg)

        # here, `X` is the total number of counts per feature regardless
        # of the region where the reads map
        self.X = self.S + self.U + self.A
        self.data_shape = self.X.shape
        assert type(self.X) == np.ndarray

        # set normalization scale for velocity fitting
        self.counts_per_cell_after = 1e4

        return

    def __getstate__(self,) -> dict:
        """
        Override the default `__getstate__` behavior
        so we do not pickle huge arrays.

        Returns
        -------
        d : dict
            object state dictionary, with large arrays removed
            to allow pickling and passage to child processes.
        
        Notes
        -----
        When we perform multiprocessing, we pickly the `VelocityCI`
        class to pass to workers. Here, we remove all large memory
        objects from the `__getstate__` method which is used during
        the pickle process to gather all the relevant components of
        an object in memory. We provide access to a shared buffer 
        with these objects to each worker to avoid copying them.
        """
        d = dict(self.__dict__)
        for attr in ['X', 'S', 'U', 'A']:
            del d[attr]
            del d[attr+'_batch']
        large_arr = ['adata', 'SUA_hat', 'embed', 'velocity_estimates']
        for k in large_arr:
            if k in d.keys():
                del d[k]
        return d

    def _sample_abundance_profile(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Given an observed mRNA abundance profile, fit a multinomial
        distribution and randomly sample a corresponding profile.

        Parameters
        ----------
        x : np.ndarray
            [Genes,] observed mRNA counts vector.

        Returns
        -------
        x_hat : np.ndarray
            [Genes,] a randomly sampled abundance profile, 
            given the multinomial distribution specified by `x`.
        """
        # we need to instantiate a local random state to ensure
        # each multiprocess thread generates true random numbers
        local_rnd = np.random.RandomState()
        # cast everything to `np.float64` before operations due to a
        # `numpy` bug
        # https://github.com/numpy/numpy/issues/8317
        x = x.astype(np.float64)
        # compute relative abundance profile as feature proportions
        pvals = x / np.sum(x)
        # sample a count distribution from the multinomial
        x_hat = local_rnd.multinomial(
            n=int(np.sum(x)),
            pvals=pvals,
        )
        return x_hat

    def _sample_spliced_unspliced(
        self,
        s: np.ndarray,
        u: np.ndarray,
        a: np.ndarray,
        x_hat: np.ndarray,
    ) -> np.ndarray:
        """Sample the proportion of spliced/unspliced reads for a 
        randomly sampled mRNA profile given observed spliced and
        unspliced read counts.

        Parameters
        ----------
        s : np.ndarray
            [Genes,] observed spliced read counts for each gene.
        u : np.ndarray
            [Genes,] observed unspliced read counts for each gene.
        a : np.ndarray
            [Genes,] ambiguous read counts for each gene.
        x_hat : np.ndarray
            [Genes,] sampled total gene counts profile.

        Returns
        -------
        sua_hat : np.ndarray
            [Genes, (Spliced, Unspliced, Ambiguous)] read counts
            randomly sampled from a multinomial.
        """
        # we need to instantiate a local random state to ensure
        # each multiprocess thread generates true random numbers
        local_rnd = np.random.RandomState()
        # Genes, (Spliced, Unspliced, Ambiguous)
        sua_hat = np.zeros((len(x_hat), 3))
        # compute total reads per feature
        x = s + u + a
        x = x.astype(np.float64)
        
        # for each gene, sample the proportion of counts that originate
        # from spliced, unspliced, or ambiguous regions using a multinomial
        # distribution parameterized with the observed proportions
        for g in range(len(x_hat)):

            if x[g] == 0:
                sua_hat[g, :] = 0
                continue

            pvals = np.array([s[g], u[g], a[g]], dtype=np.float64) / x[g]
            sua_hat[g, :] = local_rnd.multinomial(
                n=x_hat[g],
                pvals=pvals,
            )

        return sua_hat

    def _sample_cell(self,
                     i: int,
                     ) -> np.ndarray:
        """Draw samples for a single cell.

        Parameters
        ----------
        i : int
            cell index in `.X, .S, .U, .A` matrices.

        Returns
        -------
        sua_hat : np.ndarray
            [Genes, (Spliced, Unspliced, Ambig.)] for a single
            cell at index `i` in `.X`, ...

        Notes
        -----
        This implementation allows for simple parallelization with
        a map across the cell indices.
        """
        # gather the count arrays from a shared `RawArray`
        # buffer and reshape them from flat [N*M,] to array
        # [N, M] format
        X = np.frombuffer(
            var_args['X_batch'],
            dtype=np.float64,
        ).reshape(var_args['data_shape_batch'])
        S = np.frombuffer(
            var_args['S_batch'],
        ).reshape(var_args['data_shape_batch'])
        U = np.frombuffer(
            var_args['U_batch'],
            dtype=np.float64,
        ).reshape(var_args['data_shape_batch'])
        A = np.frombuffer(
            var_args['A_batch'],
            dtype=np.float64,
        ).reshape(var_args['data_shape_batch'])

        # get the read counts of each type for 
        # a single cell
        
        x = X[i, :]  # total read counts
        s = S[i, :]  # spliced read counts
        u = U[i, :]  # unspliced read counts
        a = A[i, :]  # ambiguous read counts

        # sample the relative abudance across genes
        x_hat = self._sample_abundance_profile(
            x=x,
        )
        # for each gene, sample the proportion of reads
        # originating from each type of region
        sua_hat = self._sample_spliced_unspliced(
            s=s,
            u=u,
            a=a,
            x_hat=x_hat,
        )
        return sua_hat

    def _sample_matrices(
        self,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Sample a spliced and unspliced counts matrix
        for a bootstrapped velocity vector estimation.

        Parameters
        ----------
        batch_size : int
            number of cells to sample in parallel.
            smaller batches use less RAM.

        Returns
        -------
        SUA_hat : np.ndarray
            [Cells, Genes, (Spliced, Unspliced, Ambiguous)] 
            randomly sampled array of read counts assigned
            to a splicing status.

        Notes
        -----
        `_sample_matrices` uses `multiprocessing` to parallelize
        bootstrap simulations. We run into a somewhat tricky issue
        do to the size of our source data arrays (`.X, .S, .U, .A`).
        The usual approach to launching multiple processes is to use
        a `multiprocessing.Pool` to launch child processes, then copy
        the relevant data to each process by passing it as arguments 
        or through pickling of object attributes.

        Here, the size of our arrays means that copying the large matrices
        to memory for each child process is (1) memory prohibitive and 
        (2) really, really slow, defeating the whole purpose of parallelization.

        Here, we've implemented a batch processing solution to preserve RAM.
        We also use shared ctype arrays to avoid copying memory across workers.
        Use of ctype arrays increases the performance by ~5-fold. From this, we
        infer that copying even just the minibatch count arrays across all the 
        child processes is prohibitively expensive.

        We can create shared ctype arrays using `multiprocessing.sharedctypes` 
        that allow child processes to reference a single copy of each 
        relevant array in memory.
        Because these data are read-only, we can get away with using
        `multiprocessing.RawArray` since we don't need process synchronization 
        locks or any other sophisticated synchronization.

        Using `RawArray` with child processes in a pool is a little strange.
        We can't pass the `RawArray` pointer through a pickle, so we have to
        declare the pointers as global variables that get inherited by each
        child process through use of an `initializer` function in the pool.
        We also have to ensure that our parent object `__getstate__` function
        doesn't contain any of these large arrays, so that they aren't 
        accidently pickled in with the class methods. To fix that, we modify
        `__getstate__` above to remove large attributes from the object dict.       
        """
        # [Cells, Genes, (Spliced, Unspliced, Ambiguous)]
        SUA_hat = np.zeros(
            self.X.shape + (3,)
        )
        # compute the total number of batches to use
        n_batches = int(np.ceil(self.X.shape[0]/batch_size))

        batch_idx = 0
        for batch in range(n_batches):
            end_idx = min(batch_idx+batch_size, self.X.shape[0])

            # set batch specific count arrays as attributes
            for attr in ['X', 'S', 'U', 'A']:
                attr_all = getattr(self, attr)
                attr_batch = attr_all[batch_idx:end_idx, :]
                setattr(self, attr+'_batch', attr_batch)

            # generate shared arrays for child processes
            shared_arrays = {'data_shape_batch': self.X_batch.shape}
            for attr in ['X_batch', 'S_batch', 'U_batch', 'A_batch']:
                data = getattr(self, attr)
                # create the shared array
                # RawArray will only take a flat, 1D array
                # so we create it with as many elements as
                # our desired data
                shared = multiprocessing.RawArray(
                    'd',  # doubles
                    int(np.prod(data.shape)),
                )
                # load our new shared array into a numpy frame
                # and copy data into it after reshaping
                shared_np = np.frombuffer(
                    shared,
                    dtype=np.float64,
                )
                shared_np = shared_np.reshape(data.shape)
                # copy data into the new shared buffer
                # this is reflected in `shared`, even though we're
                # copying to the numpy frame here
                np.copyto(shared_np, data)

                shared_arrays[attr] = shared

            # create a global dictionary to hold arguments
            # we pass to each worker using an initializer.
            # this is necessary because we can't pass `RawArray`
            # in a pickled object (e.g. as an attribute of `self`)
            global var_args
            var_args = {}

            # this method is called after each work is initialized
            # and sets all of the shared arrays as part of the global
            # variable `var_args`
            def init_worker(shared_arrays):
                for k in shared_arrays:
                    var_args[k] = shared_arrays[k]

            start = time.time()
            print(f'Drawing bootstrapped samples, batch {batch:04}...')
            with multiprocessing.Pool(
                    initializer=init_worker,
                    initargs=(shared_arrays,)) as P:
                results = P.map(
                    self._sample_cell,
                    range(self.X_batch.shape[0]),
                )

            # [Cells, Genes, (Spliced, Unspliced, Ambiguous)]
            batch_SUA_hat = np.stack(results, 0)
            SUA_hat[batch_idx:end_idx, :, :] = batch_SUA_hat
            batch_idx += batch_size

            end = time.time()
            print('Duration: ', end-start)

        return SUA_hat

    def _fit_velocity(
        self,
        SUA_hat: np.ndarray,
    ) -> np.ndarray:
        """Fit a deterministic RNA velocity model to the 
        bootstrapped count matrices.

        Parameters
        ----------
        SUA_hat : np.ndarray
            [Cells, Genes, (Spliced, Unspliced, Ambiguous)] 
            randomly sampled array of read counts assigned
            to a splicing status.

        Returns
        -------
        velocity : np.ndarray
            [Cells, Genes] RNA velocity estimates.
        """
        dtype = np.float64
        # create an AnnData object from a bootstrap sample
        # of counts
        boot = anndata.AnnData(
            X=SUA_hat[:, :, 0].astype(dtype).copy(),
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy(),
        )
        for i, k in enumerate(['spliced', 'unspliced', 'ambiguous']):
            boot.layers[k] = SUA_hat[:, :, i].astype(dtype)

        if self.velocity_prefilter_genes is not None:
            # filter genes to match a pre-existing velocity computation
            # this is useful for e.g. embedding in a common PC space
            # with the observed velocity
            boot = boot[:, self.velocity_prefilter_genes].copy()

        # normalize
        scv.pp.normalize_per_cell(
            boot,
            counts_per_cell_after=self.counts_per_cell_after,
        )

        # filter genes as in the embedding
        if hasattr(self, 'embed'):
            # if an embedded AnnData is provided
            # subset to genes used for the original embedding
            cell_bidx = np.array([
                x in self.embed.obs_names for x in boot.obs_names
            ])

            boot = boot[:, self.embed.var_names].copy()
            boot = boot[cell_bidx, :].copy()
            print(
                'Subset bootstrap samples to embedding dims: ',
                boot.shape,
            )
        else:
            msg = 'must providing an embedding object containing\n'
            msg += 'cells and genes to use for velocity estimation.'
            raise ValueError(msg)

        # log1p only the `.X` layer, leaving `.layers` untouched.
        scv.pp.log1p(boot)

        # fit the velocity model deterministically, following the original
        # RNA velocity publication
        scv.pp.pca(boot, use_highly_variable=False)
        scv.pp.moments(boot, n_pcs=30, n_neighbors=100)
        scv.tl.velocity(boot, mode='deterministic')

        return boot.layers['velocity']

    def bootstrap_velocity(
        self,
        n_iter: int = 100,
        embed: anndata.AnnData = None,
        velocity_prefilter_genes: list = None,
        verbose: bool = False,
        save_counts: str = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generated bootstrap estimates of the RNA velocity for 
        each cell and gene.

        Parameters
        ----------
        n_iter : int
            number of bootstrap iterations to perform.
        embed : anndata.AnnData, optional
            [Cells, Genes] experiment describing the genes of interest
            and containing a relevant embedding for projection of
            velocity vectors.
        velocity_prefilter_genes : list
            genes selected by `scv.pp.filter_genes` in the embedding object
            before normalization. often selected with `min_shared_counts`.
            it is important to carry over this prefiltering step to ensure
            that normalization is comparable to the original embedding.
        verbose : bool
            use verbose stdout printing.
        save_counts : str, optional
            save sampled count matrices to the specified path as 
            `sampled_counts_{_iter:04}.npy` with shape 
            [Sample, Cells, Genes, (Spliced, Unspliced, Ambig.)].
        **kwargs passed to `_sample_matrices()`.

        Returns
        -------
        velocity : np.ndarray
            [Sample, Cells, Genes] bootstrap estimates of RNA
            velocity for each cell and gene. 
        """
        # use genes in an embedding object if provided, otherwise
        # get the n_top_genes most variable genes
        if embed is not None:
            self.embed = embed
            embed_genes = self.embed.shape[1]
        else:
            embed_genes = self.n_top_genes

        if velocity_prefilter_genes is not None:
            self.velocity_prefilter_genes = velocity_prefilter_genes
        else:
            self.velocity_prefilter_genes = None

        # store velocity estimates for each gene
        # [Iterations, Cells, Genes]
        velocity = np.zeros((n_iter, self.embed.shape[0], embed_genes))

        for _iter in range(n_iter):
            if verbose:
                print('Beginning sampling for iteration %03d' % _iter)
                
            # sample a counts matrix
            SUA_hat = self._sample_matrices(**kwargs)

            if save_counts is not None:
                # save the raw counts sample to disk
                np.save(
                    osp.join(
                        save_counts,
                        f'sampled_counts_{_iter:04}.npy',
                    ),
                    SUA_hat,
                )

            if verbose:
                print('Sampling complete.')
                print('Fitting velocity model...')
            # fit a velocity model to the sampled counts matrix
            # yielding an estimate of velocity for each gene
            iter_velo = self._fit_velocity(
                SUA_hat=SUA_hat,
            )
            velocity[_iter, :, :] = iter_velo
            if verbose:
                print('Velocity fit, iteration %03d complete.' % _iter)

        self.velocity_estimates = velocity
        return velocity

    def bootstrap_vectors(
        self,
        embed: anndata.AnnData = None,
    ) -> np.ndarray:
        """
        Generate embedded velocity vectors for each bootstrapped sample
        of spliced/unspliced counts.

        Returns
        -------
        velocity_embeddings : np.ndarray
            [n_iter, Cells, EmbeddingDims] RNA velocity vectors
            for each bootstrap sampled set of counts in the 
            provided PCA embedding space.
        """
        if embed is not None:
            self.embed = embed

        if not hasattr(self, 'embed'):
            msg = 'must provide an `embed` argument.'
            raise AttributeError(msg)

        # copy the embedding object to use for low-rank embedding
        project = self.embed.copy()
        # remove any extant `velocity_settings` to use defaults.
        # in the current `scvelo`, using non-default settings will throw a silly
        # error in `scv.tl.velocity_embedding`.
        if 'velocity_settings' in project.uns.keys():
            project.uns.pop('velocity_settings')

        # for each velocity profile estimate, compute the corresponding
        # PCA embedding of those vectors using "direct_projection",
        # aka as standard matrix multiplication.
        #
        # the `scvelo` nearest neighbor projection method introduces
        # several assumptions that we do not wish to inherit here.
        velocity_embeddings = []
        for _iter in range(self.velocity_estimates.shape[0]):
            V = self.velocity_estimates[_iter, :, :]
            project.layers['velocity'] = V

            scv.tl.velocity_embedding(
                project,
                basis='pca',
                direct_pca_projection=True,
                autoscale=False,  # do not adjust vectors for aesthetics
            )
            velocity_embeddings.append(
                project.obsm['velocity_pca'],
            )
        velocity_embeddings = np.stack(
            velocity_embeddings,
            axis=0,
        )
        self.velocity_embeddings = velocity_embeddings
        return velocity_embeddings

    def compute_ci(self,) -> np.ndarray:
        """
        Compute confidence intervals for the velocity vector
        on each cell from bootstrap samples of embedded velocity vectors.

        Returns
        -------
        velocity_intervals : np.ndarray
            [Cells, EmbeddingDims, (Mean, Std, LowerCI, UpperCI)]
            estimates of the mean and confidence interval around the
            RNA velocity vector computed for each cell.
        """
        if not hasattr(self, 'velocity_embeddings'):
            msg = 'must run `bootstrap_vectors` first to generate vector samples.'
            raise AttributeError(msg)

        # [Cells, Dims, (Mean, SD, Lower, Upper)]
        self.velocity_intervals = np.zeros(
            self.velocity_embeddings.shape[1:] + (4,)
        )
        # for each cell, compute the mean, std, and CI for
        # each dimension in the embedding
        # this provides a hypersphere of confidence for cell state transitions
        # in the embedding space
        for j in range(self.velocity_embeddings.shape[1]):
            cell = self.velocity_embeddings[:, j, :]  # Iter, Dims
            mean = np.mean(cell, axis=0)  # Dims
            std = np.std(cell, axis=0)  # Dims
            # compute the 95% CI assuming normality
            l_ci = mean - 1.96*std
            u_ci = mean + 1.96*std
            self.velocity_intervals[j, :, 0] = mean
            self.velocity_intervals[j, :, 1] = std
            self.velocity_intervals[j, :, 2] = l_ci
            self.velocity_intervals[j, :, 3] = u_ci

        return self.velocity_intervals


##################################################
# main
##################################################


def add_parser_arguments(parser):
    """Add arguments to an `argparse.ArgumentParser`."""
    parser.add_argument(
        '--data',
        type=str,
        help='path to AnnData object with "spliced", "unspliced", "ambiguous" in `.layers`',
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='output path for velocity bootstrap samples.'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=100,
        help='number of bootstrap iterations to perform.'
    )
    return parser


def make_parser():
    """Generate an `argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        description='Compute confidence intervals for RNA velocity by molecular bootstrapping'
    )
    parser = add_parser_arguments(parser)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # load anndata
    print('Loading data...')
    adata = anndata.read_h5ad(args.data)
    print(f'{adata.shape[0]} cells and {adata.shape[1]} genes loaded.')

    # check for layers
    for k in ['spliced', 'unspliced', 'ambiguous']:
        if k not in adata.layers.keys():
            msg = f'{k} not found in `adata.layers`'
            raise ValueError(msg)

    # intialize velocity bootstrap object
    print('\nBootstrap sampling velocity...\n')
    vci = VelocityCI(
        adata=adata,
    )

    # sample velocity vectors
    velocity_bootstraps = vci.bootstrap_velocity(
        n_iter=args.n_iter,
        save_counts=args.out_path,
    )

    # save bootstrap samples to disk
    np.save(
        osp.join(args.out_path, 'velocity_bootstrap_samples.npy'),
        velocity_bootstraps,
    )
    print('Done.')
    return


if __name__ == '__main__':
    main()
