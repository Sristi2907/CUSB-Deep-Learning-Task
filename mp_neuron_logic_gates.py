import torch

# McCulloch-Pitts Neuron model using step function
def mp_neuron(inputs, weights, threshold):
    weighted_sum = torch.sum(inputs * weights)
    return 1 if weighted_sum >= threshold else 0

# Test AND gate
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

# Test OR gate
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

# Test NOT gate (1 input)
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

# Custom experiment: neuron fires when at least 1 of 3 inputs is 1
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

# Attempt XOR gate with single-layer neuron
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

# Run all tests
if _name_ == "_main_":
    test_and_gate()
    test_or_gate()
    test_not_gate()
    experiment_custom_inputs()
    test_xor_gate_attempt()