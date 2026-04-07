import torch
import hippocampalseq.utils as hseu

class MomentumGridsearch():
    def __init__(self, place_fields, spikemats, dt: float, bins: tuple, theta: bool = True):
        self.dt = torch.tensor(dt)

        if theta:
            self.decays = torch.tensor([1., 25., 50., 75., 100., 200., 300., 400., 500., 800.])
            self.diffusions = torch.round(torch.logspace(1.6, 2.6, 30), decimals=2)
            self.initial_diffusion = 10 * dt
        else:
            self.decays = torch.tensor([1., 10., 20., 40., 80., 120., 200., 400., 800., 1200., 2000., 4000.])
            self.diffusions = torch.logspace(-.3, 2.4, 25)
            self.initial_diffusion = 5 * dt
        self.bins = bins
        self.total_bins = self.bins[0] * self.bins[1]
        self.prior = torch.ones(self.total_bins) / self.total_bins

        place_fields = torch.from_numpy(place_fields)
        self.emission_probabilities = hseu.calc_poisson_emission_probabilities(
            torch.from_numpy(spikemats).double(), 
            place_fields, 
            self.dt
        )

    def _calc_o1_transition(self, diffusion):
        itrans = torch.zeros((self.total_bins, self.total_bins))
        m = torch.arange(self.bins[0])
        n = torch.arange(self.bins[1])
        mm,nn = torch.meshgrid(m, n)
        for i in range(self.bins[0]):
            for j in range(self.bins[1]):
                tt = torch.exp(
                    -(( nn - i)**2 + (mm - j)**2) / (2 * diffusion**2 * self.dt)
                ).ravel()
                itrans[:, i*self.bins[0] + j] = tt / torch.sum(tt)
        return itrans

    def _calc_o2_transition(self, diffusion, decay):
        var = (diffusion * self.dt)**2 / (2 * decay) * (1 - torch.exp(-2 *decay * self.dt))
        transition = torch.zeros((self.bins[0], self.bins[0], self.bins[0]))
        m = torch.arange(self.bins[0])
        for i in range(self.bins[0]):
            for k in range(self.bins[0]):
                mean = (1 + torch.exp(-decay * self.dt)) * k - (torch.exp(-decay * self.dt) * i)
                tt = torch.exp(-((m - mean)**2) / (2 * var))
                norm = torch.sum(tt)
                if tt == 0:
                    maxprob = 0 if mean < 0 else self.bins[0] - 1
                    tt[maxprob] = 1 
                else:
                    tt = tt / norm 
                transition[:, k, i] = tt 
        return transition

    def _forward_backward_o2(self, initial_transition, transition):
        n_states, T = self.emission_probabilities.shape
        n_states_sqrt = int(torch.sqrt(torch.tensor(n_states)))

        alphas = torch.zeros(T, n_states)
        conditionals = torch.zeros(T)
        betas = torch.zeros(T, n_states, n_states)

        alphas_0 = self.prior * self.emission_probabilities[:, 0]
        conditionals[0] = torch.sum(alphas_0)
        alphas[0] = alphas_0 / conditionals[0]

        alpha_1 = ((initial_transition * alphas_0).T * self.emission_probabilities[:,1]).T 
        conditionals[1] = torch.sum(alpha_1)
        alphas[1] = alpha_1 / conditionals[1]

        for t in range(2, T):
            alpha_t1 = alphas[t-1].reshape((n_states_sqrt, n_states_sqrt, n_states_sqrt, n_states_sqrt))
            y_sum = torch.einsum('nlj,klij->nkli', transition, alphas[t-1])
            xy_sum = torch.einsum('mki,nkli->mnkl', transition, y_sum)
            xy_sum = xy_sum.reshape((n_states, n_states))

            alpha = (xy_sum.T * self.emission_probabilities[:,t]).T
            conditionals[t] = torch.sum(alpha)
            alphas[t] = torch.sum(alpha / conditionals[t], axis=1)

        betas[-1] = torch.ones(n_states, n_states)
        for t in range(T-2, -1, -1):
            betas_t1 = betas[t+1].reshape((n_states_sqrt, n_states_sqrt, n_states_sqrt, n_states_sqrt))
            emission = self.emission_probabilities[:,t+1].reshape(
                (n_states_sqrt, n_states_sqrt)
            )
            betas_xy = betas_t1 * emission
            y_sum = torch.einsum('jln,klmn->jklm', transition.permute(2,1,0), betas_xy)
            xy_sum = torch.einsum('ikm,jklm->ijkl', transition.permute(2,1,0), y_sum)
            beta_t = xy_sum / conditionals[t]
            betas[t] = beta_t

        return alphas, conditionals, betas

    def _likelihood(self, conditionals):
        return torch.sum(torch.log(conditionals))

    def _latent_marginals(self, alphas, betas, conditionals):
        n_states, T = self.emission_probabilities.shape
        latent_marginals = torch.zeros(T, n_states)
        _alphas = alphas.clone()
        _alphas[_alphas == 0] = 10.0**-30.0
        _betas = betas.clone()
        _betas[_betas == 0] = 10.0**-30.0
        for t in range(1, T):
            marginal = torch.sum((alphas[t] * betas[t]) * conditionals[t], axis=0)
            latent_marginals[t] = marginal

        return latent_marginals 

    def _viterbi(self, initial_transition, transition):
        n_states, T = self.emission_probabilities.shape

        omegas = torch.zeros(T, n_states)
        phis   = torch.zeros(T-1, n_states)

        omegas[0] = torch.log(self.prior) + torch.log(self.emission_probabilities[:,0])
        psum = torch.log(initial_transition) + omegas[0]
        
        phis[0] = torch.argmax(psum, axis=1)
        pmax = torch.max(psum, axis=1)
        omegas[1] = torch.log(self.emission_probabilities[:,1]) + pmax

        for t in range(2, T):
            psum = torch.log(transition) + omegas[t-1]
            phis[t-1] = torch.argmax(psum, axis=1)
            pmax = torch.max(psum, axis=1)
            omegas[t] = torch.log(self.emission_probabilities[:,t]) + pmax

        z_max = torch.zeros(T)
        z_max[-1] = torch.argmax(omegas[-1])
        for t in range(T-2, -1, -1):
            z_max[t] = phis[t, int(z_max[t+1])]

        return z_max

    def run_gridsearch(self):
        o1 = self._calc_o1_transition(self.initial_diffusion)

        for diffusion in self.diffusions:
            for decay in self.decays:
                o2 = self._calc_o2_transition(diffusion, decay)
