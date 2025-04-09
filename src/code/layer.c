#include "layer.h"


void validateSize(int input_size, int output_size){
    if (input_size <= 0  || output_size <= 0){
        err(1,"input size or output size should be absolute positive value\n");
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
            return NULL; // Handle the error case appropriately
    }
}

int initialize_layer(Layer* newLayer, int input_size, int output_size, int have_bias, ActivationType activation){
    
    validateSize(input_size, output_size);

    newLayer.input_size = input_size;
    newLayer.output_size = output_size;
    newLayer.activation = activation;

    newLayer.weights = malloc(output_size*input_size*sizeof(float));
    if(newLayer.weights == NULL){
            err(1,"error during weights allocations");
            return 0;
        }

    if (have_bias){
        newLayer.biases = malloc(output_size*sizeof(float));
        if(newLayer.biases == NULL){
            err(1,"error during biases allocations");
            return 0;

        }
    }
    else{
        newLayer.biases = NULL;
    }

    //TODO check error allocations
    newLayer.input_cache = malloc(sizeof(float) * newLayer.input_size);
    newLayer.output_cache = malloc(sizeof(float) * newLayer.output_size);
    newLayer.weight_grads = malloc(sizeof(float) * newLayer.input_size * newLayer.output_size);
    newLayer.bias_grads = malloc(sizeof(float) * newLayer.output_size);
    newLayer.delta = malloc(sizeof(float) * newLayer.output_size);

    return 1;
}




void forwardInputLayer(inputLayer* inputLayer, Layer* currentLayer){
    // Find the correct activation function
    float (*activation_func)(float) = get_activation_function(currentLayer->activation_type);

    // Copy previous output into input of current layer 
    currentLayer->input_cache = inputLayer->inputs;

    // Iterate over each neuron in the current layer
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

void forwardPropagationHidden(Layer* previousLayer, Layer* currentLayer){

}
