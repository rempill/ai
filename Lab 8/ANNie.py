import numpy as np


class ANN:
    def __init__(self, input_size,hidden_size,output_size,epochs=100,learning_rate=0.01,verbose=False):
        # save architecture sizes and model options
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        # init weights with xavier initialization to handle massive inputs
        # otherwise we get exploding gradient -> when multiplying a bunch
        # of numbers we would end up with numbers too massive for sigmoid
        self.w1=np.random.randn(input_size,hidden_size) * np.sqrt(1./self.input_size)
        self.w2=np.random.randn(hidden_size,output_size) * np.sqrt(1./self.hidden_size)

        # init biases with zeros
        self.b1=np.zeros((1,self.hidden_size))
        self.b2=np.zeros((1,self.output_size))

    @staticmethod
    def sigmoid(x):
        # clip to avoid overflow for large numbers
        return 1/(1+np.exp(-np.clip(x,-15,15)))

    @staticmethod
    def sigmoid_derivative(x):
        # x is sigmoid func output
        return x*(1-x)

    def forward(self,X):
        # calc raw hidden layer values and apply activation func
        # saved a1 so we can adjust hidden layer weights later
        z1=np.dot(X,self.w1)+self.b1
        self.a1=self.sigmoid(z1)

        # output layer and prediction
        z2=np.dot(self.a1,self.w2)+self.b2
        prediction=self.sigmoid(z2)

        return prediction

    def fit(self,X,y):
        if y.ndim==1:
            y=y.reshape(-1,1)

        for epoch in range(self.epochs):
            # forward pass to get prediction
            prediction=self.forward(X)

            # backward propagation to calibrate weights
            error=y-prediction
            # gradient for output
            d_output=error*self.sigmoid_derivative(prediction)
            error_hidden=np.dot(d_output,self.w2.T)

            # gradient for hidden layer
            d_hidden=error_hidden*self.sigmoid_derivative(self.a1)

            # update output weights and bias
            self.w2+=np.dot(self.a1.T,d_output)*self.learning_rate
            self.b2+=np.sum(d_output,axis=0,keepdims=True)*self.learning_rate

            # update hidden weights and bias
            self.w1+=np.dot(X.T,d_hidden)*self.learning_rate
            self.b1+=np.sum(d_hidden,axis=0,keepdims=True)*self.learning_rate

            if self.verbose and epoch % 10 == 0:
                loss=np.mean(np.square(error))
                print("epoch:",epoch,"loss:",loss)

    def predict(self,X,treshold=0.5):
        raw_predictions=self.forward(X)
        return (raw_predictions > treshold).astype(int)

    def predict_proba(self,X):
        raw_predictions=self.forward(X)
        return raw_predictions



