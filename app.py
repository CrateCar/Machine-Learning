# neural plasticity addition experiment
# if our neural network can output values, that are added together WITHOUT using probabilities, then we can prove that our equation works
import math
import random
import json

inp = [1,2]

# loading up network

layers = {"nodes":[],"softmax_nodes":[],"weights":[],"biases":[]}
readNetwork = False

try:
    file = open("network.json","r").read()
    layers = json.loads(file)
    readNetwork = True
except:
    layers = {"nodes":[],"weights":[],"biases":[]}
    readNetwork = False

def saveNetwork(l_data):
        try:
            open("network.json","w").write(json.dumps(l_data,indent=4))
        except:
            print("Couldn't save network to 'network.json'")

# standard mathematical functions

def z(nodes,weights,biases,i,j1): # complexity: O(n)
    sum = 0
    for j2 in range(0,len(nodes[i-1])):
        sum += nodes[i-1][j2]*weights[i-1][j2][j1]
    sum += biases[i-1][j1]
    return sum

def sigmoid(x):
    return 1/(1+math.pow(math.e,-x))

def P(a,network_capacity,offset):
    return (a * network_capacity) + offset

def C(x,y):
    return math.pow((x-y),2)

# derivatives of the former functions

def sigmoid_deriv(x):
    #return math.pow(math.e,-x) / math.pow(1+(math.pow(math.e,-z)),2)
    return sigmoid(x) * (1-sigmoid(x))

def C_deriv(a,y):
    return 2*(a-y)

def find_deriv(l_data, i1,i2,j2,j1,CONNECTION_TYPE):
    derivs = []

    #i1 represents the layer in which the recursive sequence should be on
    #i2 represents the layer in which the recursive sequence will go up to (this is the layer that houses the final connection node)
    #derivatives should be weights unless its the last layer, where it will either be one or the node depending on whether it is for a bias or weight
    #rework system, may be inaccurate

    nodes = l_data["nodes"][i1]
    weights = l_data["weights"][i1-1]

    if CONNECTION_TYPE.lower().strip() == "weight":
        if i1-1 <= i2:
            derivs = l_data["nodes"][i2][j1]*l_data["weights"][i2][j1][j2]
            return derivs
        else:
            for node in range(len(nodes)):
                n_deriv = l_data["weights"][i1-1][j1][node]
                n_deriv *= find_deriv(l_data,i1-1,i2,node,j2,CONNECTION_TYPE)
                derivs.append(n_deriv)
            
            sum_deriv = 0

            for deriv in range(len(derivs)):
                sum_deriv += derivs[deriv]

            return sum_deriv
    else:
        if i1-1 <= i2:
            derivs = l_data["weights"][i2][j1][j2]
            return derivs
        else:
            for node in range(len(nodes)):
                n_deriv = l_data["weights"][i1-1][j1][node]
                n_deriv *= find_deriv(l_data,i1-1,i2,node,j2,CONNECTION_TYPE)
                derivs.append(n_deriv)
            
            sum_deriv = 0

            for deriv in range(len(derivs)):
                sum_deriv += derivs[deriv]
            
            return sum_deriv

def connection_cost(l_data,a,network_capacity,offset):
    # use a for loop maybe?  graph theory maybe?
    # connect the ending nodes to the particular, unified node
    # implement gradient descent into this same function

    for i in range(len(l_data["weights"])):
        for j2 in range(len(l_data["weights"][i])):
            for j1 in range(len(l_data["weights"][i][j2])):
                l_data["weights"][i][j2][j1] = l_data["weights"][i][j2][j1] - C_deriv(l_data["softmax_nodes"][0],(a/network_capacity)+offset)*sigmoid_deriv(l_data["softmax_nodes"][0])*(find_deriv(l_data,3,i,j1,j2,"weight"))
                
                node_updates = calculate_nodes(inp,layers)

                l_data["nodes"] = node_updates[0]
                l_data["softmax_nodes"] = node_updates[1]

    for i in range(len(l_data["biases"])):
        for j2 in range(len(l_data["biases"][i])):
            l_data["biases"][i][j2] = l_data["biases"][i][j2] - C_deriv(l_data["softmax_nodes"][0],(a/network_capacity)+offset)*sigmoid_deriv(l_data["softmax_nodes"][0])*find_deriv(l_data,3,i,0,j2,"bias")
            
            node_updates = calculate_nodes(inp,layers)

            l_data["nodes"] = node_updates[0]
            l_data["softmax_nodes"] = node_updates[1]
                
    return l_data

# network set up

def matrix_init(n): # complexity: O(n)
    unfilled_matrix = []

    for _ in range(n):
        unfilled_matrix.append(random.randrange(-100,100)/100)

    return unfilled_matrix

def layer_init(neurons,repetitions): # complexity: O(neurons * repetitions)
    unfilled_layer = []

    for _ in range(repetitions):
        unfilled_layer.append(matrix_init(neurons))

    return(unfilled_layer)


def setup(INPUT_LAYER_NEURONS,HIDDEN_LAYERS,HIDDEN_LAYER_NEURONS,OUTPUT_NEURONS): # set up for the a values, the w values, and the b values
    w_temp = [] # temporary weights matrix
    a_temp = [] # temporary nodes matrix
    b_temp = [] # temporary biases matrix

    #Node Handling
    a_temp.append(matrix_init(INPUT_LAYER_NEURONS))

    for _ in range(HIDDEN_LAYERS):
        a_temp.append(matrix_init(HIDDEN_LAYER_NEURONS))
    
    a_temp.append(matrix_init(OUTPUT_NEURONS))

    #Weight Handling
    for i in range(1,len(a_temp)):
        w_temp.append(layer_init(len(a_temp[i]),len(a_temp[i-1])))

    #Bias Handling
    for i in range(1,len(a_temp)):
        b_temp.append(matrix_init(len(a_temp[i])))

    return [a_temp,w_temp,b_temp] # returns by nodes, weights, then biases

def calculate_nodes(x,l_data): # start from the input and calculate all the nodes
    nodes = l_data["nodes"]
    weights = l_data["weights"]
    biases = l_data["biases"]

    nodes[0] = x # set first nodes to the input values
    final_nodes = []

    for i in range(1,len(nodes)): # calculate the inner values based on the input and weights
        for j1 in range(len(nodes[i])):
            nodes[i][j1] = z(nodes,weights,biases,i,j1)
    
    for j1 in range(0,len(nodes[len(nodes)-1])): # apply softmax (sigmoid function) to the last value
        final_nodes.append(sigmoid(nodes[len(nodes)-1][j1]))
    
    return [nodes,final_nodes]

setup_data = []

if not readNetwork:
    setup_data = setup(2,2,10,1)

    layers["nodes"] = setup_data[0]
    layers["weights"] = setup_data[1]
    layers["biases"] = setup_data[2]

node_updates = calculate_nodes(inp,layers)

layers["nodes"] = node_updates[0]
layers["softmax_nodes"] = node_updates[1]
#saveNetwork(layers)

node = int(P(layers["softmax_nodes"][len(layers["softmax_nodes"])-1],10,0))

print(f"{inp[0]}+{inp[1]}={node}")

print(find_deriv(layers,3,0,0,0,"weight"))

print(C(layers["nodes"][len(layers["nodes"])-1][0],(3/10)))

for i in range(5):
    layers = connection_cost(layers,3,10,0)

    node_updates = calculate_nodes(inp,layers)

    layers["nodes"] = node_updates[0]
    layers["softmax_nodes"] = node_updates[1]

    print("[TRAINING]: " + str(C(layers["softmax_nodes"][0],(3/10))))

    node = int(P(layers["softmax_nodes"][len(layers["softmax_nodes"])-1],10,0))
    print(f"{inp[0]}+{inp[1]}={node}")

inp = [2,1]

node_updates = calculate_nodes(inp,layers)

layers["nodes"] = node_updates[0]
layers["softmax_nodes"] = node_updates[1]

node = int(P(layers["softmax_nodes"][len(layers["softmax_nodes"])-1],10,0))
print(f"{inp[0]}+{inp[1]}={node}")

# solve for gradient descent, utilize DP to do so
# when training amongst different values and inputs, get an average amongst the costs and make that the step value
# generate log files each training run
