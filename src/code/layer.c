#include "layer.h"


void validateSize(int input_size, int output_size){
    if (input_size <= 0  || output_size <= 0){
        err(1,"input size or output size should be absolute positive value\n");
    }
}

//activations functions

float relu(float x) {
    return (x > 0) ? x : 0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float tanh_func(float x) {
    return tanh(x);
}


//Loss functions 

float compute_mse_loss(float* prediction, float* target, int size) {
    float squareSum = 0;
    for (int i = 0; i < size; i++) {
        float error = prediction[i] - target[i];
        squareSum += error * error;
    }

    return squareSum / size;
}

float compute_cross_entropy_loss(float* prediction, float* target, int size) {
    float epsilon = 1e-15;  // A small value to avoid log(0)
    float entropySum = 0;
    
    for (int i = 0; i < size; i++) {
        // Adding epsilon to the prediction to avoid log(0)
        entropySum += target[i] * logf(prediction[i] + epsilon);
    }
    
    return -entropySum;  // Return the negative sum of the log probabilities
}


//gradients functions

void batchGradientDescent(
    float* weights,            
    float* weight_gradients,    
    float* biases,              
    float* bias_gradients,      
    int input_size,             
    int output_size,            
    int m                       
) {
    // Update weights using the batch gradient
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights[i * input_size + j] = weights[i * input_size + j] - LEARNING_RATE * weight_gradients[i * input_size + j] / m;
        }
    }

    // Update biases using the batch gradient
    for (int i = 0; i < output_size; i++) {
        biases[i] = biases[i] - LEARNING_RATE * bias_gradients[i] / m;
    }

    
}





// Function to get the correct activation function based on the layer's activation type
float (*get_activation_function(ActivationType activation_type))(float) {
    switch (activation_type) {
        case RELU:
            return &relu;
        case SIGMOID:
            return &sigmoid;
        case TANH:
            return &tanh_func;
        default:
            err(1,"incorrect activation function type");
    }
}

int initialize_layer(Layer* newLayer, int input_size, int output_size, int have_bias, ActivationType activation){
    
    validateSize(input_size, output_size);

    newLayer->input_size = input_size;
    newLayer->output_size = output_size;
    newLayer->activation = activation;

    newLayer->weights = malloc(output_size*input_size*sizeof(float));
    if(newLayer->weights == NULL){
            err(1,"error during weights allocations");
            return 0;
        }

    if (have_bias){
        newLayer->biases = malloc(output_size*sizeof(float));
        if(newLayer->biases == NULL){
            err(1,"error during biases allocations");
            return 0;

        }
    }
    else{
        newLayer->biases = NULL;
    }

    //TODO check error allocations
    newLayer->input_cache = malloc(sizeof(float) * newLayer->input_size);
    newLayer->output_cache = malloc(sizeof(float) * newLayer->output_size);
    newLayer->weight_grads = malloc(sizeof(float) * newLayer->input_size * newLayer->output_size);
    newLayer->bias_grads = malloc(sizeof(float) * newLayer->output_size);
    newLayer->delta = malloc(sizeof(float) * newLayer->output_size);

    return 1;
}



void forwardPropagation(Layer* layers,          // Layers of the network
    int num_layers){
        
    for (int i = 1; i < num_layers-1; i++){
    Layer* previousLayer = &layers[i-1];
    Layer* currentLayer = &layers[i];

    float (*activation_func)(float) = get_activation_function(currentLayer->activation);

    currentLayer->input_cache = previousLayer->output_cache;

    for (size_t i = 0; i < currentLayer->output_size; i++) {
        float weighted_sum = 0;
        for (size_t j = 0; j < currentLayer->input_size; j++) {
            weighted_sum += currentLayer->input_cache[j] * currentLayer->weights[i * currentLayer->input_size + j];
        }

        // Add bias
        weighted_sum += currentLayer->biases[i];

        // Apply activation function and store in output_cache
        currentLayer->output_cache[i] = activation_func(weighted_sum);
    }
    }
}



void backPropagationForOneLayer(Layer *layer, float *target,int sizeTarget){
    float *delta = malloc(sizeTarget*sizeof(float));
    for (int i = 0; i < sizeTarget;i++){
        delta[i] = layer->output_cache[i] - target[i];
    }

    for(int i = 0; i < layer->output_size;i++){
        for (size_t j = 0; j < layer->input_size; j++){
            layer->weight_grads[i * layer->input_size + j] = delta[i] * layer->input_cache[j];
        }
        
    }
    layer->bias_grads = delta;
}

void backpropagate(
    Layer* layers,          // Layers of the network
    int num_layers,         // Number of layers in the network
    float* targets          // Target output (for the last layer)
) {
    // 1. Calculate error (delta) for the output layer
    int last_layer_index = num_layers - 1;
    Layer* output_layer = &layers[last_layer_index];
    float* output_cache = output_layer->output_cache; // Predicted output
    float* bias_gradients = output_layer->bias_grads;
    float* weight_gradients = output_layer->weight_grads;
    
    // Compute the delta for the output layer
    for (int i = 0; i < output_layer->output_size; i++) {
        // For MSE (Mean Squared Error) loss
        // or for cross-entropy loss, use appropriate error formula
        float error = output_cache[i] - targets[i];
        bias_gradients[i] = error;
        
        for (int j = 0; j < output_layer->input_size; j++) {
            weight_gradients[i * output_layer->input_size + j] = error * output_layer->input_cache[j];
        }
    }

    // 2. Backpropagate the error to previous layers
    for (int l = num_layers - 2; l >= 0; l--) {
        Layer* current_layer = &layers[l];
        Layer* next_layer = &layers[l + 1];
        
        // Initialize gradients for the current layer
        float* current_weight_gradients = current_layer->weight_grads;
        float* current_bias_gradients = current_layer->bias_grads;
        float* current_output_cache = current_layer->output_cache;
        float* current_input_cache = current_layer->input_cache;

        // Compute the delta for each neuron in the current layer
        for (int i = 0; i < current_layer->output_size; i++) {
            // Compute delta for current layer neurons based on next layer's gradients
            float delta = 0;
            for (int j = 0; j < next_layer->output_size; j++) {
                delta += next_layer->weight_grads[j * next_layer->input_size + i] * next_layer->delta[j];
            }
            current_layer->delta[i] = delta * activation_derivative(current_output_cache[i]);
            
            // Compute weight gradients and bias gradients
            for (int j = 0; j < current_layer->input_size; j++) {
                current_weight_gradients[i * current_layer->input_size + j] = current_layer->delta[i] * current_input_cache[j];
            }
            current_bias_gradients[i] = current_layer->delta[i];
        }
    }
}
