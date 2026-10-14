import torch
import numpy as np
import pynapple as nap
import hippocampalseq.utils as hseu
import hippocampalseq.models as hsem

def unconstrained(
        raw_freq: torch.Tensor,
        fs: torch.Tensor|float,
        *_
    ) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 2 * torch.pi / fs * raw_freq
    return theta,raw_freq

def sigmoid_clamp_constraint(
    raw_freq: torch.Tensor,
    fs: torch.Tensor|float,
    min_hz: float = 0.1,
    max_hz: float = 200.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map an unconstrained scalar/vector into a frequency in the biologically
    relevant band [min_hz, max_hz] Hz.
    In rats, biologically relevant bands are thought to be in these ranges:
    - Delta: 0.1-4 Hz
    - Theta: 4-12 Hz
    - Alpha: 7-12 Hz
    - Beta: 12-35 Hz
    - Low Gamma: 30-55 Hz
    - Mid Gamma: 55-90 Hz
    - High Gamma: 90-140 Hz
    - Ripples: 140-200 Hz
    (Buzsaki & Wang, 2012, "Mechanisms of Gamma Oscillations")

    Args:
        raw_freq (torch.Tensor): Unconstrained trainable tensor, one value per oscillator. Passed 
            through a sigmoid and clamped by the min/max allowed frequencies. Units of logit(Hz).
        fs (torch.Tensor|float): Sampling rate in samples per unit time
        min_hz (float): Lower frequency bound in Hz
        max_hz (float): Upper frequency bound in Hz

    Returns:
        (torch.Tensor): Angular frequency in radians/sample, for torch.sin/cos
        (torch.Tensor): Ordinary frequency in cycles per unit time (Hz)
    """
    frequency_hz = min_hz + (max_hz - min_hz) * torch.sigmoid(raw_freq)
    theta = 2 * torch.pi / fs * frequency_hz
    return theta, frequency_hz

def fill_missing_oscillators(
        frequencies: torch.Tensor,
        damping: torch.Tensor,
        K: int,
        frequency_range: tuple[float, float],
        default_damping: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fills in missing oscillators from the AR model initialization using a given range of frequencies.
    Linearly spans the frequency range, and uses a default damping for the missing oscillators.

    Args:
        frequencies (torch.Tensor): Frequencies of the oscillators. Works with ordinary and angular frequencies but must be 
            consistent with the frequency_range units.
        damping (torch.Tensor): Damping of the oscillators
        K (int): Number of oscillators to fill
        frequency_range (tuple[float, float]): Range (min,max) of frequencies to fill in.
        default_damping (float, optional): Default damping for the missing oscillators. Defaults to 0.9.

    Returns:
        (torch.Tensor): Filled in frequencies
        (torch.Tensor): Filled in damping
    """
    missing = K - frequencies.numel()
    if missing <= 0:
        return frequencies[:K], damping[:K]

    lo, hi = frequency_range
    extra_f = torch.linspace(
        lo, hi, missing + 2,
        dtype=frequencies.dtype,
        device=frequencies.device,
    )[1:-1]
    extra_a = torch.full_like(extra_f, default_damping)

    return (
        torch.cat((frequencies, extra_f)),
        torch.cat((damping, extra_a)),
    )

def whittle_nll(
        params,
        Y: list[torch.Tensor],
        f_s: torch.Tensor|float,
        frequency_range: tuple[float, float],
        *,
        remove_mean: bool = True,
        include_dc: bool = True
    ):
    raw_frequncies, raw_damping, component_covariances, tau2 = params
    a_k = torch.tanh(raw_damping).reshape(-1)
    f = sigmoid_clamp_constraint(
        raw_frequencies, 
        f_s,
        min_hz=frequency_range[0],
        max_hz=frequency_range[1]
    )[0].reshape(-1)
    sigma_k = torch.exp(component_covariances)**2

    if not (a_k.numel() == f.numel() == sigma_k.numel()):
        raise ValueError(
            f"Expected raw_damping, raw_frequencies, and component_covariances to have the same number of elements. "
            f"Got {damping.numel()}, {f.numel()}, and {sigma_k.numel()}"
        )
    K = a_k.numel()

    
    nll = 0.0

    c,s = torch.cos(f), torch.sin(f)
    blocks = torch.stack((
        torch.stack((c,-s), dim=-1),
        torch.stack((s, c), dim=-1),
    ), dim=-2)
    blocks = F * a_k[:,None]
    F = torch.block_diag(*blocks.unbind(dim=0))

    Q = hseu.block_diagonal_scaled_identity(sigma_k, 2)

    H = torch.zeros(2 * K, dtype=f.dtype, device=f.device)
    H[0::2] = 1.0

    complex_dtype = torch.complex128 if f.dtype == torch.float64 else torch.complex64
    F = F.to(complex_dtype)
    H = H.to(complex_dtype)
    Q = Q.to(complex_dtype)

    for y in Y:
        T = y.numel()
        if T < 2:
            raise ValueError(
                f"Expected at least 2 elements in y. Got {y.numel()}"
            )
        
        y_spec = y - y.mean() if remove_mean else y
        fft = torch.fft.rfft(y_spec)
        periodogram = torch.abs(fft) ** 2 / T

        omega = 2 * torch.pi * torch.fft.rfftfreq(T, device=y.device, dtype=y.dtype)

        eye = torch.eye(2 * K, dtype=complex_dtype, device=y.device)
        phase = torch.exp(-1j * omega).to(complex_dtype)
        A = eye.unsqueeze(0) -  hseu.atleast_3d(phase) * F.unsqueeze(0)

        G = torch.linalg.solve(A, eye.expand_as(A))
        spectral_cov = G @ Q @ G.mH

        model_spectrum = torch.einsum(
            'i,fih,j->f', H, spectral_cov, H.conj()
        ).real + tau2
        model_spectrum = torch.clamp(model_spectrum, min=torch.finfo(model_spectrum.dtype).eps)

        weights = torch.full_like(periodogram, 2.0)
        weights[0] = 1.0
        if T % 2 == 0:
            weights[-1] = 1.0

        if not include_dc:
            periodogram    = periodogram[1:]
            weights        = weights[1:]
            model_spectrum = model_spectrum[1:]

        nll += torch.sum(
            weights * (periodogram * torch.log(model_spectrum) / model_spectrum)
        )
    return nll
   

class LFPDecomposition(hsem.LinearGaussianSystem):
    r"""Implmementation of state-space model for LFP decomposition from
    Matsuda & Komaki, 2017, "Time Series Decomposition into Oscillation Components and Phase Estimation"
    The full system dynamics are:
        $$\begin{align}
            z_{k,t+1} &= a_k\begin{pmatrix}cos(2\pi f_k\Delta t) & -sin(2\pi f_k\Delta t) \\ sin(2\pi f_k\Delta t) & cos(2\pi f_k\Delta t) \end{pmatrix}
                z_{k,t} + \nu_{k,t} \\ 
            x_t &= \sum_{k=1}^K z_{k,t,1} + w_t \\
            \nu_{k,t} &\sim \mathcal{N}(0, \sigma_k^2 I_2) \\
            w_t &\sim \mathcal{N}(0, \tau^2)
        \end{align}$$
    The two components of each oscillator are the real and imaginary parts of the complex-valued oscillator state. 
    The hidden states are a set of block-diagonal matrices, allowing each oscillator to move independently. 
    The emission matrix sums the real parts of all oscillators to produce the observed signal.
    """
    def __init__(self, 
            n_components: int, 
            dt: float, 
            frequency_range: tuple[float,float] = (0.1, 200.0),
            initialization_method: str = 'autoregressive',
            objective_function: str = 'kalman'
        ):
        """Initialize the LFP decomposition model.

        Args:
            n_components (int): Number of oscillatory components to model.
            dt (float): Time step size in seconds.
            frequency_range (tuple[float,float]): Frequency range for the oscillators in Hz. Defaults to (0.1, 200.0).
            initialization_method (str): Method for initializing parameters ('uniform', 'normal', 'autoregressive'). Defaults to 'autoregressive'.
            objective_function (str): Objective function to use for fitting ('kalman', 'whittle'). Defaults to 'kalman'.
        """
        latent_dim = n_components * 2
        emission_matrix = torch.ones((1, latent_dim)) # $y_t = \sum_{k=1}^K x_{k,t,1} + w_t$
        emission_matrix[0,1::2] = 0 
        super().__init__(
            latent_dim     = latent_dim,
            emission_dim   = 1,
            order          = 1,
            default_params = {
                'initial_mean'    : torch.zeros((latent_dim,1)),
                'emission_bias'   : torch.zeros((1,1)),
                'emission_matrix' : emission_matrix, 
                'transition_bias' : torch.zeros((latent_dim,1)),
            },
            initialization_method = initialization_method
        )

        self.f_s = torch.tensor(1.0 / dt)
        self.frequency_range = frequency_range
        self.n_components = n_components
        if initialization_method == "autoregressive":
            self.initialization_method = "autoregressive"
            initialization_method = "uniform" # Reset this so we can still initialize $\tau$ and $\sigma_k$
        else:
            # Frequency constraint $0 \le 2\pi f_k \Delta t \le \pi$
            # so we apply a sigmoid and divide by $2\Delta t$
            self.raw_frequencies = self.random_initializer(
                "frequencies", 
                (n_components,1),
                self.initialization_method
            ) 
            # Amplitude must fulfill $|a_k| \lt 1$ to have PSD covariance, so we apply
            # tanh elsewhere
            self.raw_damping = self.random_initializer(
                "raw_damping", 
                (n_components,1),
                self.initialization_method
            )
            self.component_covariances = self.random_initializer(
                "component_covariances", 
                (n_components,1),
                initialization_method
            )

        self.tau2 = self.random_initializer(
            "tau", 
            (1,1),
            initialization_method
        )

        self.n_parameters = 3 * n_components + 1
        self.objective_function = objective_function

    def name(self):
        return f"{self.n_components}-dimensional Univariate LFP Decomposition"

    def _initialize_autoregression(self, Xtrain: list[torch.Tensor]):
        ar_frequencies = hsem.select_best_autoregression(
            Xtrain, 
            max_lag=max(self.n_components*2, self.n_components + 1),
        )
        phi = ar_frequencies.coefficients[0].reshape(-1)
        p = phi.numel()
        companion = phi.new_zeros((p,p))
        companion[0] = phi
        if p > 1:
            companion[1:, :-1] = torch.eye(p - 1, dtype=phi.dtype, device=phi.device)

        roots = torch.linalg.eigvals(companion)
        roots = roots[roots.imag > 1e-7]
        if roots.numel() == 0:
            raise ValueError("AR fit has no complex roots to initialize oscillators")
        
        damping = torch.clamp(torch.abs(roots), max=1-1e-6)
        theta   = torch.angle(roots)
        frequency_hz = theta / (2 * torch.pi) * self.f_s

        lo,hi = self.frequency_range
        keep = (frequency_hz >= lo) & (frequency_hz <= hi)
        frequency_hz = frequency_hz[keep]
        damping      = damping[keep]
        order        = torch.argsort(frequency_hz)
        frequency_hz = frequency_hz[order]
        damping      = damping[order]

        frequency_hz, damping = fill_missing_oscillators(
            frequency_hz, 
            damping, 
            self.n_components, 
            self.frequency_range
        )

        p = ((frequency_hz - lo) / (hi - lo)).clamp(1e-6, 1-1e-6)
        self.raw_frequencies = hseu.atleast_2d(torch.logit(p))
        self.raw_damping     = hseu.atleast_2d(torch.atanh(damping))

        var = ar_frequencies.innovation_variance[0] / self.n_components
        var = var.expand(self.n_components)
        self.component_covariances = torch.log(var) * .5
        self.component_covariances = hseu.atleast_2d(self.component_covariances)


    def _construct_transition_matrix(self, mode: str = "train") -> torch.Tensor:
        r"""Transition matrix
        $$\alpha_k\begin{pmatrix} cos(2\pi f_k\Delta t) & -sin(2\pi f_k\Delta t) \\
            sin(2\pi f_k\Delta t) & cos(2\pi f_k\Delta t) \end{pmatrix}
        $$
        """

        f = sigmoid_clamp_constraint(
            self.raw_frequencies, 
            self.f_s,
            min_hz=self.frequency_range[0],
            max_hz=self.frequency_range[1]
        )[0].squeeze()
        c,s = torch.cos(f), torch.sin(f)
        blocks = torch.stack((
            torch.stack((c,-s), dim=-1),
            torch.stack((s, c), dim=-1),
        ), dim=-2)
        blocks = blocks * torch.tanh(self.raw_damping[:,None])
        return torch.block_diag(*blocks.unbind(dim=0))

    def _construct_transition_covariance(self, mode: str = "train") -> torch.Tensor:
        r"""Transition covariance
        $$\begin{pmatrix} \sigma_k^2 & 0 \\ 0 & \sigma_k^2 \end{pmatrix}
        $$
        """
        return hseu.block_diagonal_scaled_identity(torch.exp(self.component_covariances)**2, 2)

    def _construct_emission_covariance(self, mode: str = "train") -> torch.Tensor:
        return self.tau2

    def _construct_initial_covariance(self, mode: str = "train") -> torch.Tensor:
        block_variance = torch.exp(self.component_covariances) \
            / (1 - torch.tanh(self.raw_damping)**2)
        return hseu.block_diagonal_scaled_identity(block_variance, 2)


    def _solve_parameters(
            self, 
            values: LDSResults, 
            stats: LDSStatistics, 
            optimizer: str = "Adam",
            lr: float = 0.01,
            n_epochs: int = 1000, 
            gd_tol: float = 1e-3
        ): 
        params = [
            hseu.grad_tensor(self.raw_frequencies),
            hseu.grad_tensor(self.raw_damping),
            hseu.grad_tensor(self.component_covariances)
        ]

        if self.objective_function == 'whittle':
            params += [hseu.grad_tensor(self.tau2)]

        def loss_closure(
                params, 
                n_batches: int, 
                stats: LDSStatistics, 
                f_s: torch.Tensor,
                frequency_range: tuple[float, float]
            ):
            frequencies,amplitudes,component_covariances = params

            Ez   = stats.Ez
            Ezz  = stats.Ezz
            Ezz1 = stats.Ezz1

            sigma_k = torch.exp(component_covariances)**2
            a_k = torch.tanh(amplitudes)

            # Construct transition matrix
            f = sigmoid_clamp_constraint(
                frequencies, 
                f_s,
                min_hz=frequency_range[0],
                max_hz=frequency_range[1]
            )[0].squeeze()
            c,s = torch.cos(f), torch.sin(f)
            blocks = torch.stack((
                torch.stack((c,-s), dim=-1),
                torch.stack((s, c), dim=-1),
            ), dim=-2)
            blocks = blocks * a_k[:,None]
            F = torch.block_diag(*blocks.unbind(dim=0))
            Q = hseu.block_diagonal_scaled_identity(sigma_k, 2)

            # Construct initial variance
            P1 = hseu.block_diagonal_scaled_identity(
                sigma_k / (1 - a_k**2), 
                2
            )
            
            total_loss = 0
            for i in range(n_batches):
                T = len(Ez[i])

                iloss = (
                    torch.logdet(P1)
                    + torch.trace(hseu.mulinv(P1, Ezz[i][0]))
                )

                tl1 = Ezz[i][1:]
                tl2 = Ezz1[i] @ F.mT 
                tl3 = F @ Ezz[i][:-1] @ F.mT
                tloss = torch.sum(tl1 - tl2 - tl2.mT + tl3, axis=0)
                tloss = (
                    (T - 1) * torch.logdet(Q)
                    + torch.trace(hseu.mulinv(Q, tloss))
                )

                total_loss += (iloss + tloss) / 2.0

            return total_loss
        
        if self.objective_function == 'whittle':
            objective = whittle_nll
            args = {
                'Y'               : values.observations,
                'f_s'             : self.f_s,
                'frequency_range' : self.frequency_range
            }
        elif self.objective_function == 'kalman':
            objective = loss_closure
            args = {
                'n_batches'       : len(values.observations),
                'stats'           : stats,
                'f_s'             : self.f_s,
                'frequency_range' : self.frequency_range
            }
        
        loss,params = hseu.optimize(
            objective, 
            params, 
            args,
            {
                'optimizer' : optimizer, 
                'lr'        : lr, 
                'n_epochs'  : n_epochs, 
                'gd_tol'    : gd_tol
            }
        )
        self.raw_frequencies       = params[0].detach()
        self.raw_damping           = params[1].detach()
        self.component_covariances = params[2].detach()

        # Estimate tau independently from the other variables.
        self.tau2 = self._emission_cov_mle(stats)
        self._initialize_globals()

        return loss

    def fit(self,
            Xtrain: list[torch.Tensor],
            Xvalid: list[torch.Tensor]|None = None,
            n_iter: int = 1000,
            emtol: float = 1e-3,
            patience: int = 5,
            val_freq: int = 10,
            **maximization_args
        ) -> LDSResults:
        if self.initialization_method == "autoregressive":
            Xtrain = self._initialize_observations(Xtrain)
            self._initialize_autoregression(Xtrain)
        return super().fit(
            Xtrain=Xtrain, 
            Xvalid=Xvalid, 
            n_iter=n_iter, 
            emtol=emtol, 
            patience=patience, 
            val_freq=val_freq,
            **maximization_args
        )

def extract_lfp_components(
        smoothed_means: list[torch.Tensor]
    ):
    amplitudes = []
    phases = []
    for mean in smoothed_means:
        # $x_1 + i x_2$
        x1 = mean[:,::2]
        x2 = mean[:,1::2]
        amplitudes.append(torch.sqrt(x1**2 + x2**2))
        phases.append(torch.atan2(x2, x1))
    return amplitudes, phases

def bayesian_lfp_decomposition(
        lfp_data, 
        dt: float, 
        max_components: list[int]|int = 10, 
        metric: str = "aic",
        random_seed_sequence: list[int]|None = None,
        initialization_method: str = "autoregressive"
    ):
    best_model = None
    best_values = None
    best_metric = np.inf
    if not isinstance(random_seed_sequence, list):
        random_seed_sequence = [random_seed_sequence]
    if not isinstance(max_components, list):
        max_components = list(range(1, max_components + 1))

    for seed in random_seed_sequence:
        if seed is not None:
            torch.manual_seed(seed)
        for k in max_components:
            print(f"Fitting {k}-component LFP decomposition model...")
            lfp = LFPDecomposition(k, dt, initialization_method=initialization_method)
            values = lfp.fit(lfp_data)
            if metric == "aic":
                aic = values.aic
                if aic < best_metric: 
                    best_metric = aic
                    best_model  = lfp
                    best_values = values
            elif metric == "bic":
                bic = values.bic
                if bic < best_metric: 
                    best_metric = bic
                    best_model  = lfp
                    best_values = values 

            print()
    print( f"Best model:")
    print( f"    {best_model.name()}")
    print( f"    AIC = {best_values.aic}")
    print( f"    BIC = {best_values.bic}")
    print( f"    $\omega_k$ = {sigmoid_clamp_constraint(best_model.raw_frequencies, 1 / dt)[0]} [rad/s]")
    print( f"    $f_k$ = {sigmoid_clamp_constraint(best_model.raw_frequencies, 1 / dt)[1]} [Hz]")
    print( f"    $a_k$ = {torch.tanh(best_model.raw_damping)}")
    print(fr"    $\sigma_k$ = {torch.exp(best_model.component_covariances)}")
    print(fr"    $\tau^2$ = {best_model.tau2}")

    return best_model, best_values
    