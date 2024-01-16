# Activations
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math

class RelU():
  def __call__(self, x):
    return np.where(x >= 0, x, 0)

  def gradient(self, x):
    return np.where(x >= 0, 1, 0)

class TanH():
  def __call__(self, x):
    return 2 / (1 + np.exp(-2*x)) - 1

  def gradient(self, x):
    return 1 - np.power(self.__call__(x), 2)

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
    
class MLPNetwork():
  def __init__(self, X, y ,X_val, y_val, epochs, learning_rate, neurons=128, batch_size=100, hidden_activation='RelU'):
    #data
    self.X = X
    self.y = y
    self.X_val = X_val
    self.y_val = y_val
    #HyperParameters
    self.neurons = neurons
    self.batch_size = batch_size
    self.epochs = epochs
    self.learning_rate = learning_rate
    #Activation function
    if hidden_activation == 'RelU':
      self.hidden_activation = RelU()
    else:
      self.hidden_activation = TanH()
    self.output_activation = Softmax()

    self.N = self.X.shape[0]
    #metrics
    self.train_loss = list()
    self.train_acc = list()
    self.val_loss = list()
    self.val_acc = list()
    self.metrics = [self.train_loss,self.train_acc,self.val_loss,self.val_acc]

  def cross_entropy(self, y, probabilities):
    # Clipping the values so that we avoid division by zero
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)
    loss = - y * np.log(p) - (1 - y) * np.log(1 - p)
    return loss

  def gradient(self, y, probabilities):
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)
    grad = - (y / p) + (1 - y) / (1 - p)
    return grad

  def accuracy(self, y, probabilities):
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(probabilities, axis=1)
    acc = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return acc*100

  def initialize_weights(self, X, y):
    n_samples, n_features = X.shape
    _, n_outputs = y.shape
    # Hidden layer Weights
    limit   = 1 / math.sqrt(n_features)
    self.W  = np.random.uniform(-limit, limit, (n_features, self.neurons))
    self.w0 = np.zeros((1, self.neurons))
    # Output layer Weights
    limit   = 1 / math.sqrt(self.neurons)
    self.V  = np.random.uniform(-limit, limit, (self.neurons, n_outputs))
    self.v0 = np.zeros((1, n_outputs))

  def forward_pass(self, X):
    #-------- HIDDEN LAYER --------#
    hidden_input = X.dot(self.W) + self.w0
    hidden_output = self.hidden_activation(hidden_input)
    #------- OUTPUT LAYER ---------#
    output_input = hidden_output.dot(self.V) + self.v0
    y_pred = self.output_activation(output_input)
    return y_pred


  def fit(self, X, y):
    self.initialize_weights(self.X, self.y)

    for epoch in range(1, self.epochs+1):

      #Learning Rate Decay
      if epoch%20 == 0:
        self.learning_rate = self.learning_rate/2
        print(f"learning rate decayed: {self.learning_rate}")

      #divide to batches
      shuffle = np.random.permutation(self.N)
      X_batches = np.array_split(self.X[shuffle],self.N/self.batch_size)
      y_batches = np.array_split(self.y[shuffle],self.N/self.batch_size)

      for X_batch, y_batch in zip(X_batches,y_batches):
        train_acc = 0
        train_loss = 0

        # ---------------------------------- #
        #FORWARD PASS
        # ---------------------------------- #


        #-------- HIDDEN LAYER --------#
        hidden_input = X_batch.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)
        #------- OUTPUT LAYER ---------#
        output_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_activation(output_input)


        #Training Loss and Accuracy
        train_loss += self.cross_entropy(y_batch, y_pred)
        train_acc += self.accuracy(y_batch, y_pred)

        # ---------------------------------- #
        #BACK-PROPAGATION
        # ---------------------------------- #

        #Gradient WRT output layer
        grad_wrt_output = self.gradient(y_batch, y_pred) * self.output_activation.gradient(output_input)
        grad_v = hidden_output.T.dot(grad_wrt_output)
        grad_v0 = np.sum(grad_wrt_output, axis=0, keepdims=True)
        #Gradient WRT hidden layer
        grad_wrt_hidden = grad_wrt_output.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
        grad_w = X_batch.T.dot(grad_wrt_hidden)
        grad_w0 = np.sum(grad_wrt_hidden, axis=0, keepdims=True)

        #update weights
        self.V  -= self.learning_rate * grad_v
        self.v0 -= self.learning_rate * grad_v0
        self.W  -= self.learning_rate * grad_w
        self.w0 -= self.learning_rate * grad_w0



      #Training Loss and Accuracy
      train_loss = (np.sum(train_loss)/len(X_batches))
      self.train_loss.append(train_loss)
      train_acc = (train_acc/len(X_batches))
      self.train_acc.append(train_acc)

      #Validation Loss and Accuracy
      y_pred_val = self.forward_pass(self.X_val)
      val_loss = (self.cross_entropy(self.y_val, y_pred_val)).sum(axis=1).mean()
      #val_loss = (np.sum(val_loss)/len(X_batches))
      self.val_loss.append(val_loss)
      val_acc = self.accuracy(self.y_val, y_pred_val)
      self.val_acc.append(val_acc)

      print(f"Epoch: {epoch} Training loss: {train_loss}, Training Accuracy: {train_acc}%, Validation Loss: {self.val_loss[-1]}, Validation Accuracy: {val_acc}%")


  def plot_plots(self, name):
    epochs = list(range(1, len(self.train_acc) + 1))

    #breakpoints = [i%20 == 0 for i in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, self.train_acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, self.val_acc, label='Validation Accuracy', marker='o')

    for epoch in [20, 40]:
        #if epoch == True:
        plt.axvline(x=epoch, color='red', linestyle='--', linewidth=0.8)

    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curves for {name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    ax1.plot(epochs, self.train_loss)
    ax1.set_title("Training Loss")
    ax1.set(xlabel='Epochs', ylabel='loss')

    ax2.plot(epochs, self.val_loss)
    ax2.set_title("Validation Loss")
    ax2.set(xlabel='Epochs', ylabel='Loss')