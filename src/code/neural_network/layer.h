#ifndef LAYER_H
#define LAYER_H

//library

#include <err.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LEARNING_RATE 0.01

// Enum for activation types
typedef enum { 
    RELU, 
    SIGMOID, 
    TANH, 
    LINEAR,
    SOFTMAX    
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

// Function to initialize a layer with given parameters
int initialize_layer(Layer* newLayer, int input_size, int output_size, int have_bias,int has_weight, ActivationType activation);

//FeedForward
void forwardPropagation(Layer** layers, int num_layers);


//BackPropagation
void backpropagate(
    Layer** layers,          // Layers of the network
    int num_layers,         // Number of layers in the network
    float* targets          // Target output (for the last layer)
);

//feed_forward activation functions
float relu(float x);
float sigmoid(float x);
float tanh_func(float x);

//back_propagation activation functions
float relu_derivative(float x);














//loss calculation functions
float compute_mse_loss(float* prediction, float* target, int size);
float compute_cross_entropy_loss(float* prediction, float* target, int size);



















//gradients descent functions 
void batchGradientDescent(
    float* weights,             // Current weights (size = input_size * output_size)
    float* weight_gradients,    // Gradients for the weights (size = input_size * output_size)
    float* biases,              // Current biases (size = output_size)
    float* bias_gradients,      // Gradients for the biases (size = output_size)
    int input_size,             // Number of input neurons
    int output_size,            // Number of output neurons
    int m                       // Total number of training examples);
);
void stochasticGradientDescent(
    float* weights,             // Current weights (size = input_size * output_size)
    float* weight_gradients,    // Gradients for the weights (size = input_size * output_size)
    float* biases,              // Current biases (size = output_size)
    float* bias_gradients,      // Gradients for the biases (size = output_size)
    int input_size,             // Number of input neurons
    int output_size             // Number of output neurons
);
void miniBatchGradientDescent (
    float* weights,             // Current weights (size = input_size * output_size)
    float* weight_gradients,    // Gradients for the weights (size = mini_batch_size * input_size * output_size)
    float* biases,              // Current biases (size = output_size)
    float* bias_gradients,      // Gradients for the biases (size = mini_batch_size * output_size)
    int input_size,             // Number of input neurons
    int output_size,            // Number of output neurons
    int mini_batch_size         // Number of samples in the mini-batch
);




void printLayerInfo(Layer* layer);
void free_layer(Layer* Layer);


#endif // LAYER_H
