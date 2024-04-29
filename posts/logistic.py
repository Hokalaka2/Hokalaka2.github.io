import torch

class LinearModel:

    #our model stores current and previous w
    def __init__(self):
        self.w = None 
        self.pw = None

    #scores for current w
    def score(self, X):
        if self.w is None: 
            self.w = (torch.rand(X.size()[1])-0.5) / X.size()[1]        
        return torch.matmul(X, self.w)

    #Our model predicts what X is
    def predict(self, X):
        s = self.score(X)
        return torch.where(s > 0, 1.0, 0.0)

class LogisticRegression(LinearModel):

    ## sigmoid function
    def sigmoid(self, score):
        sig = (1.0/(1.0+torch.exp(-score)))

        #since a sig of 0.0 or 1.0 causes inf or NaN values we want these to instead get just above or below that
        sig[sig == 0.0] = 0.00001
        sig[sig == 1.0] = 0.99999
        return sig

    #Empirical Loss function
    def loss(self, X, y):
        score = self.score(X)
        sig = self.sigmoid(score)
        empirical_risk = (-y * torch.log(sig)) - ((1 - y) * torch.log(1 - sig))
        return torch.mean(empirical_risk)
    
    #Empirical Gradient Loss function
    def grad(self, X, y):
        score = self.score(X)
        sig = self.sigmoid(score)
        v = (sig - y)
        v_ = v[:, None]
        grad = (X * v_)
        return torch.mean(grad, dim = 0)

        
class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model
    
    #steps through one iteration of gradient descent returns loss
    def step(self, X, y, alpha, beta):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        
        #creates previous w value
        if self.model.pw== None:
            self.model.pw = torch.rand((X.size()[1]))

        curr = self.model.w - alpha * self.model.grad(X, y) + beta * (self.model.w - self.model.pw)
        self.model.pw = torch.clone(self.model.w)
        self.model.w = curr
        return self.model.loss(X, y)