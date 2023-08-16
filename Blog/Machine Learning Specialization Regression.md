# OpenAI Machine Learning Specialization: Regression and Classification

The first course I am taking to start this journey is the Stanford & OpenAI **Supervised Machine Learning: Regression and Classification** course on Coursera taught by Andrew Ng. While researching for which course to take, this one came up frequently as one of the best. It’s geared towards beginners and covers some popular algorithms used in machine learning. The course is the first of three in a Machine Learning specialization offered from Coursera that I plan to complete as well. ([https://www.coursera.org/specializations/machine-learning-introduction](https://www.coursera.org/specializations/machine-learning-introduction)?)

[This course](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction) focused solely on introducing Regression and Classification learning models. It took me about a week to complete, the estimated hours to completion Coursera gives you is veryyy generous. It does not take as long as what is indicated on the course website. Anyways let’s dive into what I learned.

### Supervised Learning

First, what is Supervised learning?

Supervised learning is a machine learning technique where the algorithm learns to predict output variables based on input variables and labeled training data. In other words, the algorithm is trained using a dataset where the correct outputs are already known, and it uses this information to make predictions for new, unseen data. The goal of supervised learning is to minimize the difference between the predicted output and the actual output for each observation in the training data.

So, in Supervised learning models we need two data sets to train it. One is the input data and another is the correct output data. In simple terms, you plug the input data into the model, let the model make a prediction based on an algorithm that is defined in code, then compare the model’s prediction with the right answer for that specific input data. You then have to tweak the model’s learning algorithm based on how right/wrong it’s predicted answer was. This is how the model “learns”. Pretty similar to how humans learn with flashcards!

The difference between regression and classification models is that regression models produce answers that can very, while classification models will only provide specific answers. The classic example is using a ML model to identify if an image is a Dog or Cat. There are only two options so that is a classification problem. Whereas predicting the price of a Home for sale or temperature outside are regression problems because the range of possible answers are practically infinite.

### Prediction

We kind of glossed over the important steps in that general description of how Supervised learning works, so let’s unpack it a little. I mentioned that you plug some input data into the model and it comes up with a prediction. Well, how does it make that prediction? It uses an algorithm with parameters that it can tweak as it “learns” to change the output. The first learning algorithm that was covered in the course was Linear Regression.

Linear Regression is a supervised learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or multiple predictor variables (also called independent variables or features). The algorithm assumes that there is a linear relationship between the predictor variables and the outcome variable.

The key idea behind linear regression is to find the best-fitting straight line that minimizes the distance between the actual data points and the predicted values on the line. The equation of a simple linear regression model can be expressed as:

y=mx+b

Where:

- y is the dependent variable (the one you're trying to predict).
- x is the independent variable (the input or predictor variable).
- m is the slope of the line, representing the change in y for a unit change in x.
- b is the y-intercept, the value of y when x is 0.

Linear regression can also be extended to represent a model with multiple parameters, called multiple linear regression:

y=b0+b1x1+b2x2+…+bnxn

These represent the parameters change with each iteration while the model “learns” and are sometimes called weights and biases.

For each entry in the training set, the model will run the linear regression algorithm using its current parameters and the features on the input data. The features are simply the different types of input data each entry can have.

That is how the prediction is made with Linear Regression!

### Comparison

Once the model makes a prediction, it’s time to compare that prediction to the actual correct answer. This is where something called a **Cost Function** comes into play. The primary purpose of a cost function is to measure how well a model's predictions match the true outcomes, and it serves as a basis for optimizing or training the model. The cost function basically tells the model how wrong it was. The cost function helps guide the process of finding the optimal parameters for a model. These parameters are adjusted iteratively in an attempt to minimize the cost function, which essentially means making the model's predictions as close as possible to the actual data.

A common cost function used in Linear regression is the least squares error function which is calculated as the average of the squared differences between the model’s prediction and the correct answer.

- *MSE=1/2*∑(true−predicted)**2**

The goal is to tweak or fit the model’s prediction parameters in a fashion that minimizes the cost function, which means the model’s prediction is more accurate.

### Learning

With the cost function, we now know the error in the model’s predictions. Now it’s time for the model to learn! The learning process takes place by attempting to tweak the models parameters to reduce the Cost Function as much as possible. We need another algorithm to do this. A common one is **Gradient Descent**.

Gradient descent is an optimization algorithm used to find the values of the model's parameters that minimize the cost function. The "gradient" refers to the slope or the rate of change of the cost function with respect to each parameter. In simple terms, the gradient indicates the direction in which the cost function increases most rapidly.

The basic idea of gradient descent is to start with an initial guess for the parameters and then iteratively update them in the opposite direction of the gradient. This update process continues until the algorithm converges to a set of parameter values that correspond to a local minimum of the cost function.

When using gradient descent, you start with an initial set of parameters for your model. In each iteration, you calculate the gradient of the cost function with respect to each parameter. This gradient provides the direction in which you should adjust the parameters to decrease the cost function.

The update rule for each parameter is typically of the form:

new parameter=old parameter−learning rate×gradient

where the learning rate is a hyperparameter that determines the step size taken in each iteration.

By repeatedly applying the gradient descent updates, the model's parameters gradually move toward values that correspond to lower and lower values of the cost function. The process continues until convergence, which is when the changes in the parameter values become very small or when a predefined number of iterations is reached.

he gradient of the cost function provides the necessary information for updating the model's parameters in a way that leads to better predictions and reduced prediction errors.

### Learning Rate

The learning rate controls how big of a step the algorithm takes when updating w & b.

Learning rate is always a positive number.

At a local minima, the slope of a tangent line on the point w is equal to 0. The derivative of J(w) evaluates to 0.

Keeping learning rate fixed, we can reach the local minimum. The derivative of J(w) naturally gets smaller and smaller the closer we get to a minimum value. Thus the equation evaluates to a smaller and smaller step.

The learning rate is a function of how much the parameters change between iterations.

new parameter=old parameter−learning rate×gradient

If the cost value goes up and then down, it’s a clear sign that the gradient descent algorithm is not working (bug or learning rate is too large)

with a small enough learning rate, the cost function should decrease on every iteration