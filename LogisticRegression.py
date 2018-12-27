'''
Logistic Regression - Algorithm Implementation

_______________________________________________________________________________

     Paper : Towards Concise Models of grid Stability
Authored by: Vadim Arzamasov, Klemens Bohm, Patrick Jochem
    country: Germany
_______________________________________________________________________________

To deal with the fixed inputs and equality issues, the authors investigate
system stability for different design points and apply one
specific data-mining method, namely logistic regression, to the results.

To deal with the many inputs and to make the description
of stability regions more comprehensible, we replace the inputs
of the system with aggregates, referred to as features.

'''

# Importing the required dependencies
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Logistic_Regression:
    
    def intercept(self, x):
        m = x.shape[0]
        intercept = np.ones([m,1])
        self.x = np.concatenate((intercept,x), axis=1)
 
    '''
       Function: intercept()
    Description: The function adds an additional column matrix to the feature matrix in order to obtain the required intercept
                 
    '''

    def __init__(self):

        self.number_of_iterations = 10000
        self.learning_rate = 0.01

    '''
       Function: __init__()
    Description: Constructor to initialize the number of times gradient is iterated in the gradient descent function for the given learning rate
    
    '''

    def sigmoid(self, z):
        return 1/ ( 1 + np.exp(-1*z) )

    '''
       Function: sigmoid()
    Description: Function to calculate sigmoid value for a given value z.
    
    '''

    def cost_function(self,h,y):
        J = sum((-1*y*np.log(h))- ( (1-y)* np.log(1-h) ) ) / y.shape[0]
        return J
    
    '''
       Function: cost_function()
    Description: calculates the loss for the given parameters and target variables
    
    '''

    def learning_curve(self):
        counter_1 = 0
        scale = 1
        n = np.zeros([self.number_of_iterations+1,1])
        for i in np.nditer(n):
            n[counter_1] = scale
            scale += 1
            counter_1 += 1
        plt.plot(n, self.cost, 'r-')
        plt.xlabel('No.of.iterations')
        plt.ylabel('Cost Function')
        plt.title('Learning Curve')
        plt.show()

    '''
       Function: learning_curve()
    Description: Function to plot the graph between No.of iterations and Cost function, this can be used to evalute optimization of gradient descent 
    
    '''

    def predict(self,x_test,y_test):
        h = np.dot(x_test, self.theta.T)
        pred = self.sigmoid(h)
        counter_1 = 0

        for i in np.nditer(pred):
            if(i>=0.5):
                pred[counter_1] = 1
            else:
                pred[counter_1] = 0
            counter_1 += 1
        self.accuracy(y_test, pred)

    '''
       Function: predict()
    Description: Function to predict from the given features after the model is trained
    
    '''
    
    def accuracy(self, y_test, pred):
        accuracy = accuracy_score(y_test,pred) * 100
        print('\nAccuracy of the model= ',accuracy, '%' )

    '''
       Function: accuracy()
    Description: It calculates the accuracy of the model using the predicted value and true value
    
    '''

    def gradient_descent(self):
        counter_1 = 0
        for i in tqdm( range (self.number_of_iterations), desc='Training the model', leave= True ):
            z = np.dot(self.x, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.x.T, (h-self.y)) / self.y.shape[0]
            self.theta -= self.learning_rate * gradient 
            counter_1 += 1
            self.cost[counter_1] = self.cost_function(h,self.y)

    '''
       Function: gradient_descent()
    Description: Peroforms gradient descent in the given feature matrix and target matrix
    
    '''

    def fit(self,x,y):
        self.cost = np.ones([self.number_of_iterations+1,1])
        self.theta = np.zeros(x.shape[1])
        self.x = x
        self.y = y
        self.gradient_descent()
    
    '''
       Function: fit()
    Description: This function trains the logistic regression model
    
    '''

def main():        
    data = pd.read_csv('Data_for_UCI_named.csv') 
    test = pd.read_csv('data_test.csv')

    x = data.iloc[:,:-2].values
    y = data.iloc[:,-1].values
    x_test = test.iloc[:,:-2].values
    y_test = test.iloc[:,-1].values
    retry = 'y'
    while retry == 'y':
        print('\n\n\t\t\tL O G I S T I C  R E G R E S S I O N')
        print('\n\t\t1. Algorithm Implementation')
        print('\n\t\t2. Scikit learn')
        user = int(input('\n\t\tChoice: '))
        if ( user == 1 ):
            clf = Logistic_Regression()
            clf.fit(x,y)
            clf.predict(x_test, y_test)
            clf.learning_curve()
        elif ( user == 2 ):
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit(x,y)
            pred = clf.predict(x_test)
            print(pred)
            accuracy = accuracy_score(y_test,pred) * 100
            print('\n\t\tAccuracy: ', accuracy)
        else:
            print('\n\t\t\I N V A L I D')
            exit()
        retry = input('\n\t\tRetry (y/n): ')
if __name__ == "__main__":
    main()

