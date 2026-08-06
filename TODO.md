# TODO

## General
- [ ] Finalize incorporating Eryn's code
    - [ ] Predecoding check
        - [x] Determines common recording interval of LFP, spikes, and position
        - [x] Detects and filters out theta cycles in each segment (hide gaps)
        - [x] Assign an oscillation ID to each position
        - [x] Segments it into run segments 
        - [ ] Filters out segments that are too long (Check this? 160ms seems really short)
        - [x] Filter out non-monotonic phase cycles
        - [x] Count # spikes occuring in major/minor windows of theta
        - [x] Creates spikemats (T,N)
    - [ ] Theta sequence decoding error metrics
    - [ ] Modality-specific decoding
- [x] Separate out theta and replay processing 
- [ ] Write replay processing
- [ ] 1D linear decoding checks
- [ ] Clean up plotting code
- [ ] Pipeline plotting and analysis code
- [ ] Document various preprocessing functions
- [ ] Ask Brad for Janni Open field 1,2 data

## Momentum model
- [x] Check coordinate systems for all incoming and internal data.
- [ ] Simulate trajectory and spiking code to test model recovery
- [x] Double-check linear track data

## Circuit subspace model
- [ ] Add covariance and mean scaling from sum of spikes spikes for replay sequences
    - Comes from $\tau_E$ ion the network model
- [ ] Finish testing the $$\dot{z} = v_t + u(x_t - z_t) + \sigma \xi_t$$ model
    - $x_t$ is the true position here
- [ ] Incorporate LFP data