# TODO

## General
- [ ] Finalize incorporating Eryn's code
    - [ ] Modality-specific decoding
- [ ] Write replay processing
- [ ] Pipeline plotting and analysis code
- [ ] Document various preprocessing functions

## Momentum model
- [ ] Simulate trajectory and spiking code to test model recovery

## Circuit subspace model
- [ ] Add covariance and mean scaling from sum of spikes spikes for replay sequences
    - [ ] Double-check initial step from model.
    - [ ] Try 2D u value
    - Comes from $\tau_E$ ion the network model
- [ ] Finish testing the $$\dot{z} = v_t + u(x_t - z_t) + \sigma \xi_t$$ model
    - $x_t$ is the true position here
- [ ] Incorporate LFP data

## Modeling tasks
- [ ] Try 2D synaptic input in CANNDynamics
- [ ] Use emission convariance multiplied by a constant in the CANNDynamics model
- [ ] Use likelihood position as feedforward input in CANNDynamics
- [x] Use likelihood position's velocity as input into MomentumVelocityBias