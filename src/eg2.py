######################
# Processing irregular data is sometimes a little finickity.
# With neural CDEs, it is instead relatively straightforward.
#
# Here we'll look at how you can handle:
# - irregular sampling
# - missing data
# - variable-length sequences
#
# In every case, the only thing that needs changing is the data preprocessing. You won't need to change your model at
# all.
#
# Note that there's little magical going on here -- the way in which we're going to prepare the data is actually 
# pretty similar to how we would do so for an RNN etc.
######################

import torch
import torchcde


######################
# We begin with a helper for solving a CDE over some data.
######################

def _solve_cde(x):
    # x should be a tensor of shape (..., length, channels), and may have missing data represented by NaNs.

    # Create dataset
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
    print(coeffs.size())

    # Create model
    input_channels = x.size(-1)
    hidden_channels = 4  # hyperparameter, we can pick whatever we want for this
    output_channels = 10  # e.g. to perform 10-way multiclass classification

    class F(torch.nn.Module):
        def __init__(self):
            super(F, self).__init__()
            # For illustrative purposes only. You should usually use an MLP or something. A single linear layer won't be
            # that great.
            self.linear = torch.nn.Linear(hidden_channels,
                                          hidden_channels * input_channels)

        def forward(self, t, z):
            batch_dims = z.shape[:-1]
            return self.linear(z).tanh().view(*batch_dims, hidden_channels, input_channels)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.initial = torch.nn.Linear(input_channels, hidden_channels)
            self.func = F()
            self.readout = torch.nn.Linear(hidden_channels, output_channels)

        def forward(self, coeffs):
            X = torchcde.CubicSpline(coeffs)
            X0 = X.evaluate(X.interval[0])
            z0 = self.initial(X0)
            zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
            zT = zt[..., -1, :]  # get the terminal value of the CDE
            return self.readout(zT)

    model = Model()

    # Run model
    return model(coeffs)


######################
# Okay, now for the meat of it: handling irregular data.
######################

def irregular_data():
    ######################
    # Begin by generating some example data.
    ######################

    # Batch of three elements, each of two channels. Each element and channel are sampled at different times, and at a
    # different number of times.
    t1a, t1b = torch.rand(7).sort().values, torch.rand(5).sort().values
    t2a, t2b = torch.rand(9).sort().values, torch.rand(7).sort().values
    t3a, t3b = torch.rand(8).sort().values, torch.rand(3).sort().values
    x1a, x1b = torch.rand_like(t1a), torch.rand_like(t1b)
    x2a, x2b = torch.rand_like(t2a), torch.rand_like(t2b)
    x3a, x3b = torch.rand_like(t3a), torch.rand_like(t3b)
    # Overall this has irregular sampling, missing data, and variable lengths.

    ######################
    # We begin by putting handling each batch element individually. Here we handle the problems of irregular sampling
    # and missing data.
    ######################

    def process_batch_element(ta, tb, xa, xb):
        # First get all the times that the batch element was sampled at, across all channels.
        t, sort_indices = torch.cat([ta, tb]).sort()
        # Now add NaNs to each channel where the other channel was sampled.
        xa_ = torch.cat([xa, torch.full_like(xb, float('nan'))])[sort_indices]
        xb_ = torch.cat([torch.full_like(xa, float('nan')), xb])[sort_indices]
        # Add observational masks
        maska = (~torch.isnan(xa_)).cumsum(dim=0)
        maskb = (~torch.isnan(xb_)).cumsum(dim=0)
        # Stack (time, observation, mask) together into a tensor of shape (length, channels).
        return torch.stack([t, xa_, xb_, maska, maskb], dim=1)

    x1 = process_batch_element(t1a, t1b, x1a, x1b)
    x2 = process_batch_element(t2a, t2b, x2a, x2b)
    x3 = process_batch_element(t3a, t3b, x3a, x3b)

    # Note that observational masks can of course be omitted if the data is regularly sampled and has no missing data.
    # Similarly the observational mask may be only a single channel (rather than on a per-channel basis) if there is
    # irregular sampling but no missing data.

    ######################
    # Now pad out every shorter sequence by filling the last value forward. The choice of fill-forward here is crucial.
    ######################

    max_length = max(x1.size(0), x2.size(0), x3.size(0))

    def fill_forward(x):
        return torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))])

    x1 = fill_forward(x1)
    x2 = fill_forward(x2)
    x3 = fill_forward(x3)

    ######################
    # Batch everything together
    ######################
    x = torch.stack([x1, x2, x3])

    ######################
    # Solve a Neural CDE: this bit is standard, and just included for completeness.
    ######################

    zT = _solve_cde(x[:,:,1:3])
    return x, zT

x, zT = irregular_data()