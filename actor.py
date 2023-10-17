import numpy as np

class ControllerMPC:
    def __init__(self, i, data, mean, variance, optimizer, running_objective, action_bounds=None, prediction_horizon = 5, discount_factor = 0.5):
        super().__init__()
        self.dim_output = 1
        self.prediction_horizon = prediction_horizon
        self.discount_factor=discount_factor
        self.optimizer = optimizer
        self.action = np.zeros(self.dim_output)
        self.action_sequence = np.squeeze(np.tile(self.action, (1, self.prediction_horizon)))
        self.action_bounds = np.tile(action_bounds.T, (self.prediction_horizon))
        self.running_objective = running_objective
        self.mean = mean
        self.variance = variance
        self.i = i
        self.data = data

    def mpc_objective(
        self,
        action_sequence
    ):

        action_sequence_reshaped = np.reshape(action_sequence, [self.prediction_horizon, self.dim_output]).T
        means = np.reshape(self.mean.loc[self.i], [self.prediction_horizon, self.dim_output]).T
        variances = np.reshape(self.variance.loc[self.i], [self.prediction_horizon, self.dim_output]).T
        volume = action_sequence_reshaped*(1 +means)/((1+action_sequence_reshaped) + action_sequence_reshaped*(1+means))
        volume = np.column_stack([self.data.Volume.loc[self.i], volume[:,:self.prediction_horizon-1]])
        observation_sequence_predicted = np.array((means, variances, volume)).squeeze()


        actor_objective = 0

        for k in range(self.prediction_horizon):
            actor_objective += self.discount_factor**k * self.running_objective(
                observation_sequence_predicted[:, k], action_sequence_reshaped[:, k], self.i
            )
        return actor_objective

    def compute_action(
        self
    ):
        self.action_sequence = self.optimizer.optimize(lambda action: self.mpc_objective(action), self.action_sequence, self.action_bounds)
        return self.action_sequence[:1]
