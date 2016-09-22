    # Training one layer neural networks
    # Procesure should be:
    # 1. Load data X
    # 2. Initialize hidden layer weights [784 * 100], bias [1 * 100]
    # 3. Initialize output layer weights [100 * 10], bias [1 * 10]
    # 4. Forward propagation. Get the final output [data * 10]
    # 5. Define loss function. Calculate loss.
    # 6. Calculate derivatives for output pre-activation
    # 7. calculate derivatives for each W and b
    # 8. update W and b
    # repeat 4-8 as a training epoch
    # 9. When testing, use the same W and b, do forward propagation.