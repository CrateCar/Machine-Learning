import random
import math

initial = 5
weight = .3265
bias = -1.523

def softmax(z):
    return 1 / (1+(math.e**-z))

def deriv_soft(z):
    return (math.pow(math.e,-z)) / math.pow(((1+(math.pow(math.e,-z)))),2)

def preNode(i,W,b):
    return (i*W)+b

def cost(x,y):
    return (x-y)**2

def deriv_cost(x,y):
    return 2*(x-y)

output = softmax(preNode(initial,weight,bias)).real
desired_output = 1

print(output)

print(cost(output,desired_output))

old_bias = bias
old_weight = weight
difference = weight
difference_bias = bias

old_z = preNode(initial,weight,bias)

while cost(output,desired_output) >= 0.00005:
    output = softmax(preNode(initial,weight,bias))
    weight = old_weight - (0.001 * (deriv_cost(output,desired_output)*deriv_soft(old_z)*(initial)))
    bias = old_bias - (0.001 * (deriv_cost(output,desired_output)*deriv_soft(old_z)))
    old_weight = weight
    old_bias = bias

print(cost(output,desired_output))
print((softmax(preNode(initial,weight,bias))))
print(bias)
print(weight)
