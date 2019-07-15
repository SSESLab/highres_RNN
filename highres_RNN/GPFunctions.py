import gptools

def fit_gp_hyperparam(x_train, y_train, x_test):
    k = gptools.SquaredExponentialKernel(param_bounds=[(0, 1e3), (0, 100)])
    gp = gptools.GaussianProcess(k)
    gp.add_data(x_train, y_train)
    gp.optimize_hyperparameters()
    y_star, err_y_star = gp.predict(x_test)

    return y_star, err_y_star


