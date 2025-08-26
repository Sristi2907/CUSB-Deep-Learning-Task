import torch


def mp_neuron(inputs, weights, threshold):
    weighted_sum = torch.sum(inputs * weights)
    return 1 if weighted_sum >= threshold else 0

def test_and_gate():
    print("AND Gate")
    weights = torch.tensor([1.0, 1.0])
    threshold = 2.0
    inputs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]
    for x1, x2 in inputs:
        input_tensor = torch.tensor([x1, x2], dtype=torch.float32)
        output = mp_neuron(input_tensor, weights, threshold)
        print(f"{x1} AND {x2} = {output}")
    print()
    
def test_or_gate():
    print("OR Gate")
    weights = torch.tensor([1.0, 1.0])
    threshold = 1.0
    inputs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]
    for x1, x2 in inputs:
        input_tensor = torch.tensor([x1, x2], dtype=torch.float32)
        output = mp_neuron(input_tensor, weights, threshold)
        print(f"{x1} OR {x2} = {output}")
    print()


def test_not_gate():
    print("NOT Gate")
    weights = torch.tensor([-1.0])
    threshold = -0.5
    inputs = [0, 1]
    for x in inputs:
        input_tensor = torch.tensor([x], dtype=torch.float32)
        output = mp_neuron(input_tensor, weights, threshold)
        print(f"NOT {x} = {output}")
    print()


def experiment_custom_inputs():
    print("Custom Gate: Fires if at least 1 of 3 inputs is 1")
    weights = torch.tensor([1.0, 1.0, 1.0])
    threshold = 1.0
    inputs = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1)
    ]
    for x1, x2, x3 in inputs:
        input_tensor = torch.tensor([x1, x2, x3], dtype=torch.float32)
        output = mp_neuron(input_tensor, weights, threshold)
        print(f"Inputs: {x1}, {x2}, {x3} → Output: {output}")
    print()


def test_xor_gate_attempt():
    print("Attempting XOR Gate with Single M-P Neuron")
    weights = torch.tensor([1.0, 1.0])
    threshold = 1.0
    inputs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]
    expected = [0, 1, 1, 0]
    for (x1, x2), exp in zip(inputs, expected):
        input_tensor = torch.tensor([x1, x2], dtype=torch.float32)
        output = mp_neuron(input_tensor, weights, threshold)
        print(f"{x1} XOR {x2} → Output: {output} (Expected: {exp})")
    print()

    print("XOR cannot be modeled with a single M-P neuron.")
    print("Reason: XOR is *not linearly separable*. A single neuron can only learn linearly separable functions like AND or OR.\n"
          "XOR requires at least one hidden layer in a neural network (i.e., multi-layer perceptron).\n")


if _name_ == "_main_":
    test_and_gate()
    test_or_gate()
    test_not_gate()
    experiment_custom_inputs()
    test_xor_gate_attempt()



#A single McCulloch-Pitts neuron is a linear classifier and can only model linearly separable functions like AND, OR, NOT.
#XOR is not linearly separable, as its true values (0,1) and (1,0) are not separable from (0,0) and (1,1) using a straight line in 2D.   
#It requires multiple neurons (layers) or non-linear functions.
