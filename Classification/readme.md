Problem 2 Classification (60 points)
For this assignment you are asked to implement both binary and multiclass classification using gradient descent to update weight and bias in each iteration. PLEASE EDIT bm_classify.py ONLY. Any other changes that cause system issues may lead to 0 point in this part.

Q2.1 Binary Classification - Perceptron vs. Logistic (28 points)
In our lecture we have been discussing how to do binary classification using different loss. Now it is your time to implement perceptron loss and logistic loss for binary classification. In this problem you are given training set  D={(xn,yn)n=1N},yi∈{0,1}∀i=1...N.  Please note that the labels are not +1 or -1 as we discussed in lectures, so think carefully before you apply formula to this question. Your task is to learn a model  wTx+b  that minimize the given loss. Please find the gradients of the two loss functions by yourself and apply average gradient descent to update  w,b  in each iteration. For perceptron loss we define  Z=yn(wTxn+b)>0  as correctly classified data.

(8 points) TODO 1 For perceptron loss that is find the minimizer of
F(w,b)=∑n=1NLperceptron(yn(wTxn+b))=∑n=1NMAX(0,−yn(wTxn+b))
 
(8 points) TODO 2 For logistic loss that is find the minimizer of
F(w,b)=∑n=1NLlogistic(yn(wTxn+b))=∑n=1Nln(1+e−yn(wTxn+b))
 
(4 points) TODO 3 Also you will find out it is convenient to use sigmoid fuction  σ(z)=(1+e−z)−1  for logistic loss, so please complete it. You can use this function in TODO 2.

(4 points for each)TODO 4 TODO 5 After you learn the models, how do you know it is good enough? The intuitive way is to make some predictions to see if those predicted results are correct or not. Here we want you complete the prediction functions. It will be like something greater than 0 and something put into sigmoid function greater than 0.5. You may find out an interesting fact here.

Q2.2 Multiclass classification - SGD vs. GD (32 points)
Well done! Ready to take our next challenge? Let's get into multiclass classification. In this question you are going to build a model to classify data into more than just two classes. Also you are going to implement both  SGD  and  GD  for multiclass classification and compare performances of the two approaches. Training dataset are similar to question Q2.1, but  yi∈{0,1,...,C−1}∀i=1...N.  Your task is to learn models for multiclass classification based on minimizing logistic loss.

Here is a short review of  SGD .

From the lecture we know multiclass logistic loss is
F(W)=∑n=1Nln(1+∑k≠yne(wk−wyn)Txn).
 
Here,  wk  is the weight vector for class  k  and  wyn  is the weight vector for class  yn .  k  is in range [0, C-1]. Now we try to apply  SGD . First we randomly pick a data  xn  and minimize logistic loss
g(W)=ln(1+∑k≠yne(wk−wyn)Txn).
 
And then find the derivative  ▽wg(W) , where  ▽wg(W)  is a  C x D  matrix.

Let's look at each row k.

If  k≠yn :
▽wkg(W)=e(wk−wyn)Txn1+∑k′≠yne(wk′−wyn)TxnxTn=P(k|xn;W)xTn
 
else:
▽wkg(W)=−∑k′≠yne(wk′−wyn)Txn1+∑k′≠yne(wk′−wyn)TxnxTn=(P(yn|xn;W)−1)xTn
 
where  P  is softmax function.

In the end, our update for  W  is
W←W−η⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢P(y=0|xn;W)⋮P(y=yn|xn;W)−1⋮P(y=C−1|xn;W)⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥xTn
 
That is the whole idea of  SGD  of logistic loss for multiclass classification.

(8 points) TODO 6 Complete the  SGD  part in def multiclass_train, and don't forget the bias  b . To randomly pick a data from dataset, you can use np.random.choice one time in each iteration.

(16 points) TODO 7 Complete the  GD  part in def multiclass_train. Compare to  SGD ,  GD  does not randomly pick a data  xn . Instead,  GD  considers all training data points to compute derivative. Please think about how to compute  GD , and again we want average gradient descent. Also there is a tricky point. When dataset is large,  GD  will takes a large amount of time. How to reduce the time? Make sure you use numpy programming instead of nested for loops, otherwise you will not finish your test on Vocareum within the time limit.
