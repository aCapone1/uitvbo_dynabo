from src.objective_functions_LQR import *
import numpy as np
from scipy.signal import dlsim, dlti, lsim, lti
from scipy.linalg import solve_discrete_are, expm, solve_continuous_are
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
from torch import nn
# from src.TVBO import TimeVaryingBOModel
from torch.autograd import Variable

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def get_prior_mean(XT: torch.Tensor, spatio_dimensions: int = 1, scaling_factors: torch.Tensor = 1, mean: torch.Tensor = 0, stddev: torch.Tensor = 1, noise=True) -> nn.Module:
    x = XT[..., 0:spatio_dimensions]*scaling_factors
    t = XT[..., -1]
    
    Q = np.eye(4) * 10
    R = np.eye(1)
    sample_time = 0.02
    # XPLUS = []
    XPLUS = np.array([])
    XU = np.array([])
    # XTU = []

    if len(t) < len(x):
        t = torch.ones_like(x)[:, 0] * t

    for xi, ti in zip(x, t):
        controller = xi.numpy()
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)
        K = get_opt_state_controller(model, Q, R)
        updated_controller = np.append(K[0, 0:4-spatio_dimensions], controller).reshape(1, -1)
        statet, inputt = perform_simulation_and_get_states(model, updated_controller, ti.numpy(), noise)
        statet = np.float32(statet)
        inputt = np.float32(inputt)
        if not (np.isnan(statet).any() or np.isinf(statet).any()):
            if XU.shape[0] == 0:
                XU = np.hstack([statet, inputt.T])[:-1,:]
                XPLUS = np.hstack([statet])[1:,:]
            else:    
                XU = np.vstack([XU,(np.hstack([statet, inputt.T])[:-1,:])])
                XPLUS = np.vstack([XPLUS, (np.hstack([statet])[1:,:])])
        
        

    # training_hor = 500
    # # XTU = torch.tensor(XTU)[:,:training_hor,:]
    # XU = torch.tensor(XU)[:training_hor,:]
    # XPLUS = torch.tensor(XPLUS)[:training_hor,:]
    XU = torch.tensor(XU)
    XPLUS = torch.tensor(XPLUS)

    # pred_model = linearRegression(inputSize=XTU.shape[-1], outputSize=XPLUS.shape[-1])
    pred_model = linearRegression(inputSize=XU.shape[-1], outputSize=XPLUS.shape[-1])

    # if torch.cuda.is_available():
    #     pred_model.cuda()

    lr = 0.001
    nepochs = 10000

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr)

    for _ in range(nepochs):
        # Converting inputs and labels to Variable
        # if torch.cuda.is_available():
        #     inputs = Variable(torch.from_numpy(XTU).cuda())
        #     labels = Variable(torch.from_numpy(XPLUS).cuda())
        # else:
        inputs = Variable(XU)
        labels = Variable(XPLUS)

        optimizer.zero_grad()
        outputs = pred_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # update parameters
        optimizer.step()

    for param in pred_model.parameters():
        param.requires_grad = False

    # return Custom_Prior_Mean(pred_model=pred_model, scaling_factors=scaling_factors)
    # simulate_predicted_system(pred_model, torch.tensor(updated_controller), 1000, torch.tensor(Q), torch.tensor(R))
    return Custom_Prior_Mean(pred_model=pred_model, spatio_dimensions=spatio_dimensions, 
                             scaling_factors=scaling_factors, Q=Q, R=R, K=K, mean=mean, stddev=stddev)


class Custom_Prior_Mean(nn.Module):
    def __init__(self, pred_model, spatio_dimensions, scaling_factors, Q, R, K = None, mean=0, stddev=1, Ts = 1000):
        super().__init__()
        self.spatio_dimensions = spatio_dimensions
        self.scaling_factors = scaling_factors
        self.pred_model = pred_model
        self.Ts = Ts
        self.K = K
        self.Q = Q
        self.R = R
        self.mean = mean
        self.stddev = stddev

    def forward(self, inputs) -> torch.Tensor:

        inputs = inputs.squeeze()[..., 0:self.spatio_dimensions]*self.scaling_factors
        
        K = torch.tensor(self.K)

        if len(inputs.shape) == 2:
            controller = torch.zeros(inputs.shape[0], K.shape[-1])    
        elif len(inputs.shape) == 3:
            controller = torch.zeros(inputs.shape[0], inputs.shape[1], K.shape[-1])
        else:
            raise NotImplementedError
        controller[..., 4-self.spatio_dimensions:] = inputs
        controller[...,0:4-self.spatio_dimensions:] = K[:, 0:4-self.spatio_dimensions]
        controller = controller.double()

        if len(inputs.shape) == 2:
            val = torch.zeros(inputs.shape[0],)
            for i, c in enumerate(controller):
                val[i] = simulate_predicted_system(self.pred_model, c, self.Ts, self.Q, self.R) 
        elif len(inputs.shape) == 3:
            val = torch.zeros(inputs.shape[0],inputs.shape[1])
            for i, batch_controllers in enumerate(controller):
                for j, c in enumerate(batch_controllers):
                    val[i, j] = simulate_predicted_system(self.pred_model, c, self.Ts, self.Q, self.R)
        else:
            raise NotImplementedError
        return (val-self.mean)/self.stddev


def get_prior_mean_4D(x: torch.Tensor, t: torch.Tensor, noise=True) -> torch.Tensor:
    sample_time = 0.02
    x_mat = []
    u_mat = []
    if len(t) < len(x):
        t = torch.ones_like(x)[:, 0] * t

    for xi, ti in zip(x, t):
        controller = xi.numpy()
        test_params = get_params(ti)
        model = get_linearized_model(test_params, sample_time)
        updated_controller = controller.reshape(1, -1)
        xt, ut = perform_simulation_and_get_states(model, updated_controller, ti.numpy(), noise)
        x_mat.append(xt)
        u_mat.append(ut)
    x_mat = torch.tensor(x_mat, dtype=torch.float)
    u_mat = torch.tensor(u_mat, dtype=torch.float)
    return x_mat, u_mat

def perform_simulation_and_get_states(model, controller, t, noise):
    obj_params = get_params(t)
    sample_time = 0.02
    simulation_time = 20

    # define weights
    Q = np.eye(4) * 10
    R = np.eye(1)

    # start algo
    if not USE_DGL:
        sim_time, states, inputs = simulate_system(model, controller, obj_params, simulation_time, sample_time, t,
                                                   noise)
    else:
        states, inputs = simulate_system_DGL(inv_pendulum, controller, obj_params, simulation_time, sample_time)

    return states, inputs



def simulate_predicted_system(pred_model, K, Ts, Q, R):
    # initial condition
    z0 = [4, 0, 0.1, -0.01]
    zt = torch.tensor(z0, dtype=float)
    states = torch.tensor(zt.reshape(-1,1).double())
    
    for _ in range(Ts):
        ut = -torch.matmul(K, zt)
        ztplus = pred_model(torch.hstack([zt, ut]).float())
        zt = ztplus.double()
        states = torch.hstack([states, ztplus.reshape(-1,1)])
    
    valid_entries = torch.logical_and(~states.isnan(), ~states.isinf()).any(dim=0)
    states = states.T[valid_entries].double().T
    inputs = - K @ states

    return get_lqr_cost_torch(states, inputs, torch.tensor(Q), torch.tensor(R))



def get_lqr_cost_torch(x, u, Q, R, eval_logcost=True):
    # get size
    dim_x = 4
    dim_u = 1

    x = x.reshape(-1, 1, dim_x)
    x_T = x.reshape(-1, dim_x, 1)
    u = u.reshape(-1, 1, dim_u)
    u_T = u.reshape(-1, dim_u, 1)
    cost = x @ Q @ x_T + u @ R @ u_T

    time_steps = u.shape[0]
    cost = torch.sum(cost) / time_steps
    return cost




# class TimeVaryingBOModelWNNPrior(TimeVaryingBOModel):

#     def run_TVBO_Wprior(self, n_initial_points, time_horizon, safe_name='', trial=0):

#         self.time_horizon = time_horizon
#         train_x, train_y, t_remain = self.generate_initial_data(n_initial_points, trial)
#         self.prior_mean = get_prior_mean_2D(train_x, self.spatio_dimensions, self.scaling_factors)

#         self.run_TVBO(n_initial_points, time_horizon, safe_name='', trial=0)

    
