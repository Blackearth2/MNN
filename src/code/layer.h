#ifndef LAYER_H
#define LAYER_H

//library

#include <err.h>


// Enum for activation types
typedef enum { 
    RELU, 
    SIGMOID, 
    TANH, 
    LINEAR 
} ActivationType;

// Structure representing a neural network layer
typedef struct {
    int input_size;      // Number of neurons in the previous layer
    int output_size;     // Number of neurons in this layer

    float *weights;      // Weights of the layer (required)
    float *biases;       // Biases of the layer (optional, NULL if unused)

    ActivationType activation;  // Activation function used for this layer

    // For training (backpropagation)
    float *input_cache;     // Input values saved during the forward pass
    float *output_cache;    // Output values after activation in the forward pass
    float *weight_grads;    // Gradients of the weights (for backpropagation)
    float *bias_grads;     // Gradients of the biases (for backpropagation)
    float *delta;           // Error term (delta) used in backpropagation
} Layer;

typedef struct {
    int output_size;

    float *inputs;
}inputLayer;

// Function to initialize a layer with given parameters
int initialize_layer(Layer* newLayer, int input_size, int output_size, int have_bias, ActivationType activation);

#endif // LAYER_H
