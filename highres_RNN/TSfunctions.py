import numpy
import matlab.engine
import math

from bayes_opt import BayesianOptimization

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


#Start MATLAB engine
eng = matlab.engine.start_matlab()
eng.cd(r'~/Documents/MATLAB/MATLAB/MATLAB_webproject')


def convert_to_matlab_double(X):

    X_out = numpy.asarray(X)
    X_out = matlab.double(X_out.tolist())

    return X_out


def run_bayes(x, design_mat, comp_mat, T0, T_h0, T_c0, H_d):

    #at first let us extract out Q_h and Q_c

    Q_h_max = x[0]
    Q_c_max = x[1]



    def target(q_h, q_c):


        flow_param = numpy.asarray([q_h, q_c])
        flow_param = matlab.double(flow_param.tolist())

        H_out, T_tank, T_h, T_c = eng.fun_TSmain01b(design_mat, comp_mat, flow_param, T0, T_h0, T_c0,
                                                    nargout=4)

        print "H_demand (fun): ", H_d
        print "H_out (fun): ", H_out

        if H_out < H_d:
            loss_out = -((H_out - H_d) / H_d) ** 2
        else:
            loss_out = -3.0 * ((H_out - H_d) / H_d) ** 2


        return loss_out


    #Here the optimzation part goes in
    gp_params = {"alpha": 1e-5}


    bo = BayesianOptimization(target, {'q_h': (0, Q_h_max),
                                       'q_c': (0, Q_c_max)})

    bo.maximize(n_iter=10, **gp_params)



    return bo.res['max']['max_params']



###Instantiate
def predict_GP(X_t, Y_t, X_e):
    kernel = RBF()
    gp = GaussianProcessRegressor(n_restarts_optimizer=5, normalize_y=True)
    gp.fit(X_t, Y_t)
    y_pred, sigma = gp.predict(X_e, return_std=True)

    return y_pred, sigma


def fix_flow_rates(Q_h_opt, Q_c_opt, Q_h_max, Q_c_max):
    # Comditions to fix hack

    if Q_h_opt < 0:
        Q_h_opt = 0.0001

    if Q_c_opt > Q_h_max:
        Q_h_opt = Q_h_max

    if Q_c_opt < 0:
        Q_c_opt = 0.0001

    if Q_c_opt > Q_c_max:
        Q_c_opt = Q_c_max


    return Q_h_opt, Q_c_opt



def fix_params(x, x_min, x_max):

    x_new = x.copy()
    for i in range(0, len(x)):
        if x[i] < x_min[i]:
            x_new[i] = x_min[i]
        elif x[i] > x_max[i]:
            x_new[i] = x_max[i]
        else:
            pass


    return x_new


def compute_Hst(H_demand, E_gen, power_to_heat, epsilon):

    H_gen = E_gen/power_to_heat
    H_st = H_demand - H_gen
    H_st = epsilon*H_st

    return H_st


def run_bayes02(x, design_mat, comp_mat, T0, T_h0, T_c0, H_rec, H_d):

    #at first let us extract out Q_h and Q_c
    T_tank = numpy.asarray(T0)
    T_tank = T_tank.flatten()
    T_top = T_tank[0]

    if T_top < x[0]:
        T_hmin = x[0]
    else:
        T_hmin = T_top


    T_hmax = x[1]
    Q_c_max = x[2]
    gamma_max = x[3]



    def target(gamma, T_hin, q_c):


        H_supTS = gamma*H_rec
        opt_param = numpy.asarray([T_hin, q_c])
        opt_mat = matlab.double(opt_param.tolist())


        H_supTS = float(H_supTS)

        [H_out, H_st, T_tank, T_hot, T_cold] = eng.fun_TSmain01c(design_mat, comp_mat, opt_mat, T0, T_h0, T_c0, H_supTS, nargout=5)

        H_supBldg = (1-gamma)*H_rec + H_out

        print "H_demand (fun): ", H_d
        print "H_supTS (fun): ", H_supTS
        print "H_supBldg (fun): ", H_supBldg
        print "H_out (fun): ", H_out
        print "H_st (fun): ", H_st

        if H_supBldg < H_d:
            mu = 0
        else:
            mu = 1.3


        kappa = 0.05
        epsilon = 0.2

        if H_supBldg > H_d and H_d >H_rec:
            #loss_out = H_supBldg/H_d + mu*H_st/H_d - kappa*Q_c_max/20.0
            loss_out = -epsilon -math.fabs(H_supBldg - H_d)/ H_d + mu * H_st / H_d - kappa * Q_c_max / 20.0

        else:
            #loss_out = 0.2*(H_supBldg/H_d + mu*H_st/H_d - kappa*Q_c_max/20.0)
            loss_out = -math.fabs(H_supBldg - H_d) / H_d + mu * H_st / H_d - kappa * Q_c_max / 20.0




        return loss_out


    #Here the optimzation part goes in
    gp_params = {"alpha": 1e-5}

    bo = BayesianOptimization(target, {'gamma': (0, gamma_max),
                                     'T_hin': (T_hmin, T_hmax), 'q_c': (0.0, Q_c_max)})

    if H_rec > H_d:
        bo.explore({'gamma': [gamma_max], 'T_hin': [T_hmax], 'q_c': [0]})
    else:
        bo.explore({'gamma': [0.0, 0, 0.0, 0.0], 'T_hin': [T_hmin, T_hmin, T_hmin, T_hmin], 'q_c': [2.0, 5.0, 10, Q_c_max]})


    bo.maximize(n_iter=10, **gp_params)



    return bo.res['max']['max_params']