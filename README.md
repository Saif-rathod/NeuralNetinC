# XOR Neural Network in C

1. **Project Overview**: A neural network in C that learns to compute the XOR function using a two-layer architecture.
2. **Architecture**: 2 input neurons → 2 hidden neurons (sigmoid) → 1 output neuron (sigmoid), trained via backpropagation.
3. **Training Details**: Uses 10,000 epochs, a learning rate of 0.1, and Mean Squared Error minimization.
4. **Usage**:
   - Compile: `gcc xor_network.c -o xor_network -lm`
   - Run: `./xor_network`
5. **Expected Output**: Final weights, biases, and accuracy progress printed to console, demonstrating XOR function learning.

MIT License.
