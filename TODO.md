# TODO

## General
- [ ] Finalize incorporating Eryn's code
- [ ] Clean up plotting code
- [ ] Pipeline plotting and analysis code
- [ ] Document various preprocessing functions
- [ ] Ask Brad for Janni Open field 1,2 data

## Momentum model
- [ ] Simulate trajectory and spiking code to test model recovery
- [ ] Double-check linear track data

## Circuit subspace model
- [ ] Add covariance and mean scaling from sum of spikes spikes for replay sequences
    - Comes from $\tau_E$ ion the network model
- [ ] Finish testing the $$\dot{z} = v_t + u(x_t - z_t) + \sigma \xi_t$$ model
    - $x_t$ is the true position here
- [ ] Incorporate LFP data