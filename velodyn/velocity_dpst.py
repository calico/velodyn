"""Compute a change in pseudotime for each cell"""
import numpy as np
import anndata
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


class dPseudotime(object):
    """Compute a change in pseudotime value for each cell
    in a single cell experiment.

    Attributes
    ----------
    adata : anndata.AnnData
        [Cells, Genes] single cell experiment.
    use_rep : str
        representation to use for predicting pseudotime coordinates.
        `adata.obsm[f'X_{use_rep}']`, `adata.obsm[f'velocity_{use_rep}']`
        must be present.
    pseudotime_var : str
        scalar variable in `adata.obs` encoding pseudotime coordinates.
    model : sklearn.neighbors.KNeighborsRegressor
        k-nearest neighbors regression model for pseudotime prediction.
    X : np.ndarray
        [Cells, Embedding] observed coordinates in embedding space.
    V : np.ndarray
        [Cells, Embedding] velocity vectors in embedding space.
    y : np.ndarray
        [Cells,] pseudotime coordinates.
    X_pred : np.ndarray
        [Cells, Embedding] predicted future coordinates.
    pst_pred : np.ndarray
        [Cells,] pseudotime coordinates inferred for positions `X_pred`.
    dpst : np.ndarray
        [Cells,] change in pseudotime coordinate.

    Methods
    -------
    _fit_model
    predict_dpst
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        use_rep: str = 'pca',
        pseudotime_var: str = 'dpt_pseudotime',
    ) -> None:
        """Compute a change in pseudotime value for each cell
        in a single cell experiment.

        Parameters
        ----------
        adata : anndata.AnnData
            [Cells, Genes] single cell experiment.
        use_rep : str
            representation to use for predicting pseudotime coordinates.
            `adata.obsm[f'X_{use_rep}']`, `adata.obsm[f'velocity_{use_rep}']`
            must be present.
        pseudotime_var : str
            scalar variable in `adata.obs` encoding pseudotime coordinates.

        Returns
        -------
        None.
        """
        self.adata = adata
        self.use_rep = use_rep
        self.pseudotime_var = pseudotime_var

        # check that necessary matrices are present
        if f'X_{use_rep}' in self.adata.obsm.keys():
            self.X = self.adata.obsm[f'X_{use_rep}']
        else:
            msg = f'X_{use_rep} is not in `adata.obsm'
            raise ValueError(msg)

        if f'velocity_{use_rep}' in self.adata.obsm.keys():
            self.V = self.adata.obsm[f'velocity_{use_rep}']
        else:
            msg = f'velocity_{use_rep} is not in `adata.obsm'
            raise ValueError(msg)

        if pseudotime_var in self.adata.obs.columns:
            self.y = self.adata.obs[pseudotime_var]
        else:
            msg = f'{pseudotime_var} is not in `adata.obs'
            raise ValueError(msg)

        return

    def _fit_model(
        self,
        n_neighbors: int = 50,
        weights: str = 'distance',
    ) -> None:
        """Fit a regression model to predict pseudotime coordinates
        from the specified embedding.

        Parameters
        ----------
        n_neighbors : int
            number of neighbors to use for regression model.
        weights : str
            method to weight neighbor contributions.
            passed to `sklearn.neighbors.KNeighborsRegressor`.

        Returns
        -------
        None. assigns `self.model`, `self.cv_scores`.
        """
        # initialize a simple kNN regressor with multiprocessing
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1,
        )

        # perform cross-validation scoring
        self.cv_scores = cross_val_score(
            self.model,
            self.X,
            self.y,
            cv=5,
        )
        print('Cross-validation scores for prediction model:')
        print(self.cv_scores)
        print('Mean : ', np.mean(self.cv_scores))
        print()

        # fit the final model on all the data
        self.model.fit(self.X, self.y)
        return

    def predict_dpst(
        self,
        step_size: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        """Predict a change in pseudotime coordinate for each cell
        in the experiment.

        Parameters
        ----------
        step_size : float
            step size to use for future cell state predictions.
            the RNA velocity vector is scaled by this coefficient
            before addition to the current position.
            we recommend step sizes smaller than `1`.
        **kwargs are passed to `self._fit_model`.

        Returns
        -------
        dpst : np.ndarray
            [Cells,] change in pseudotime value predicted for each
            cell.
        Also sets `self.pst_pred`, `self.dpst` atttributes.

        See Also
        --------
        self._fit_model
        """
        self._fit_model(**kwargs)

        # the predicted new pseudotime coordinate is the current
        # coordinate + the velocity vector, scaled by a step size
        self.X_pred = self.X + step_size * self.V
        # we predict the new coordinate's pseudotime position
        self.pst_pred = self.model.predict(self.X_pred)
        # the \Delta pseudotime coordinate is the difference between
        # predicted and observed coordinates
        self.dpst = self.pst_pred - self.y
        return self.dpst
