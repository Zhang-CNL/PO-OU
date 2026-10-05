import torch
import hippocampalseq.utils as hseu
from dataclasses import dataclass 
from .statespace import * 

@dataclass 
class ARResults:
    coefficients        : list[torch.Tensor]
    intercepts          : list[torch.Tensor]
    innovation_variance : list[torch.Tensor]
    aic                 : list[torch.Tensor]
    bic                 : list[torch.Tensor]

class ARModel(StateSpace):
    r"""Autoregressive statespace model.

    $$x_t = \sum_j \theta_j x_{t-j-1} + \epsilon$$
    """
    def __init__(self, order: int, fit_intercept: bool = True, method: str = "PARCOR"):
        assert method in ["PARCOR", "LSTSQ"]
        self.order = order
        self.method = method
        self.fit_intercept = fit_intercept
        self.n_parameters = order + 1
        if self.fit_intercept:
            self.n_parameters += 1

    def name(self):
        return f"AR({self.order}) model"

    def _parcor_to_ar(self, parcor: torch.Tensor):
        """Converts PARCOR coefficients into standard AR coefficients"""
        ar = parcor.new_empty(0)
        for k in parcor:
            if ar.numel() == 0:
                ar = k.reshape(1)
            else:
                # Levinson recursion
                ar = torch.cat((ar - k * ar.flip(0), k.reshape(1)))
        return ar

    def _ar_residuals(self, x: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
        r"""Residuals for
             $$x_t = \sum_j \theta_j x_{t-j-1} + \epsilon
             $$
        """
        windows = x.unfold(0, self.order + 1, 1)
        predictors = windows[:, :-1].flip(1)  # columns: x[t-1], ..., x[t-p]
        targets = windows[:, -1]
        return targets - predictors @ ar

    def _loglikelihood(self, residuals: torch.Tensor):
        n = residuals.numel()
        variance = torch.mean(residuals**2)
        variance = torch.clamp(variance, min=torch.finfo(residuals.dtype).eps)
        return -.5*n * (
            torch.log(torch.tensor(2*torch.pi))
            + 1 
            + torch.log(variance)
        )

    def _fit_ar(self, x: torch.Tensor):
        x = x.reshape(-1)
        if not 1 <= self.order < x.numel():
            raise ValueError("self.order must satisfy 1 <= self.order < len(x)")

        windows = x.unfold(0, self.order + 1, 1)
        X_lags = windows[:, :-1].flip(1)
        y = windows[:, -1]

        if self.method == "LSTSQ":
            if self.fit_intercept:
                X = torch.cat((torch.ones_like(y[:,None]), X_lags), dim=1)
                beta = torch.linalg.lstsq(X,y).solution
                intercept = beta[0]
                ar = beta[1:]
            else:
                beta = torch.linalg.lstsq(X_lags, y).solution
                ar = beta
                intercept = 0.
        else:
            raw_parcor = torch.zeros((self.order,), requires_grad=True)
            intercept = torch.zeros(1, requires_grad=self.fit_intercept)
            params = [raw_parcor]
            if self.fit_intercept:
                params.append(intercept)
            optimizer = torch.optim.LBFGS(
                params,
                lr=.01,
                max_iter=1000,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                ar_candidate = self._parcor_to_ar(torch.tanh(raw_parcor))
                residuals = y - intercept - X_lags @ ar_candidate
                variance = torch.mean(residuals**2)
                variance = torch.clamp(variance, torch.finfo(x.dtype).eps)
                loss = 0.5 * len(y) * torch.log(variance)
                loss.backward()
                return loss

            optimizer.step(closure)
            ar = self._parcor_to_ar(torch.tanh(raw_parcor)).detach()

        residuals = y - intercept - X_lags @ ar
        innovation_variance = torch.mean(residuals**2)
        innovation_variance = torch.clamp(innovation_variance, min=torch.finfo(residuals.dtype).eps)

        loglike = self._loglikelihood(residuals)

        return (
            loglike.detach(),
            intercept.detach(),
            ar.detach(),
            innovation_variance.detach()
        )

    def fit(self,
            X: list[torch.Tensor]
        ):
        coefficients = []
        intercepts = []
        innovation_variance = []
        aics = []
        bics = []
        for x in X:
            x = hseu.ensure_torch(x)
            ll,intercept,coeffs,var = self._fit_ar(x)
            aic = self.aic(ll)
            bic = self.bic(ll, len(x))
            coefficients.append(coeffs)
            intercepts.append(intercept)
            innovation_variance.append(var)
            aics.append(aic)
            bics.append(bic)
        return ARResults(
            coefficients,
            intercepts,
            innovation_variance,
            aics,
            bics
        )

def select_best_autoregression(
        data: list[hseu.NDArray], 
        max_lag: int, 
        criterion: str = "aic",
        method: str = "PARCOR",
        fit_intercepts: bool = False
    ):
    print("Selecting best AR model")
    best_metric = torch.inf 
    best_values = None
    for lag in range(1, max_lag + 1):
        model = ARModel(lag, fit_intercepts, method)
        values = model.fit(data)
        # Can't meaningfully compare aic/bic from different datasets
        # TODO: Figure out how to select/average over different sessions
        if criterion == "aic":
            for aic in values.aic:
                if aic < best_metric:
                    best_metric = aic
                    best_values = values 
        elif criterion == "bic":
            for bic in values.bic:
                if bic < best_metric:
                    best_metric = bic 
                    best_values = values 

    print(f"Best model:")
    print(f"    AR({len(best_values.coefficients[0])}) model")
    print(f"    Intercept: {best_values.intercepts}")
    print(f"    Coefficients: {best_values.coefficients}")
    return best_values


