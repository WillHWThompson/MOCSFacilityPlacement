import numpy as np
#calculate the covariance of two values
def cov(x,y):
        x_hat = np.mean(x)
        y_hat = np.mean(y)
        return np.sum((x-x_hat)*(y-y_hat))/(len(x)-1)
#return the variance
def variance(x):
    return cov(x,x)
#calculate the coefficent of determination
def coef_det(x,y):
    return cov(x,y)/np.sqrt(variance(x)*variance(y))
#calculate the b_value
def b_val(x,y):
    return np.sqrt(variance(y)/variance(x))


class LinReg:
    def __init__(self):
        self.x_emp = None#empirical x values
        self.y_emp = None#empirical y values 
        self.beta = None
        self.intercept = None
        self.rho = None
        self.b = None

    #calculates the intercept 
    def calc_intercept(self):
        return np.mean(self.y_emp)-self.beta*np.mean(self.x_emp)
        
    def fit(self,x,y):
        #save the empirical values
        self.x_emp = x
        self.y_emp = y
        #calculate gradient 
        self.rho = coef_det(x,y) 
        self.b = b_val(x,y)
        self.beta = self.calc_beta()#the slope of the regression
        self.intercept = self.calc_intercept()#the intercept of the regression

    def predict(self,X):
        Y = self.beta*X+self.intercept
        return Y 
        

    
class LeastSquares(LinReg):
     #calculate the gradient,slope of the function
    def calc_beta(self):
        return self.rho*self.b

    def gradient_confidence_range(self,q):
        upper_conf_range = self.rho*self.b*(1+q)
        lower_conf_range = self.rho*self.b*(1-q)
        return (upper_conf_range,lower_conf_range)


class ReducedMajorAxis(LinReg):
    #calculate the gradient,slope of the function
    def calc_beta(self):
        return np.sign(self.rho)*self.b

    def gradient_confidence_range(self,q):
        upper_conf_range = np.sign(self.rho)*self.b*(1+q/1-q)**0.5
        lower_conf_range = np.sign(self.rho)*self.b*(1-q/1+q)**0.5
        return (upper_conf_range,lower_conf_range)
