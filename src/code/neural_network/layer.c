#include "layer.h"


void validateSize(int input_size, int output_size){
    if (input_size < 0  || output_size < 0){
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

//derivated_activation functions
float relu_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

float sigmoid_derivative(float x) {
    float sig = 1.0f / (1.0f + expf(-x));  // Sigmoid function
    return sig * (1 - sig);  // Sigmoid derivative
}

float tanh_derivative(float x) {
    float tanh_x = tanhf(x);  // Tanh function
    return 1.0f - tanh_x * tanh_x;  // Tanh derivative
}

float linear_derivative(void) {
    return 1.0f;  // Derivative of a linear function is always 1
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

float (*get_derivated_activation_function(ActivationType activation_type))(float) {
    switch (activation_type) {
        case RELU:
            return &relu_derivative;
        case SIGMOID:
            return &sigmoid_derivative;
        case TANH:
            return &tanh_derivative;
        default:
            err(1,"incorrect derivated activation function type");
    }
}

void initialize_biases(float* biases, int size){
    for(int i = 0; i < size; i++){
        biases[i] = 1.0;
    }
}

float randInRange(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

void initialize_weight(float* weights, int size){
    for(int i = 0; i < size; i++){
        weights[i] = randInRange(-1.0f, 1.0f);
    }
}

int initialize_layer(Layer* newLayer, int input_size, int output_size, int have_bias,int have_weight, ActivationType activation){
    
    validateSize(input_size, output_size);


    newLayer->input_size = input_size;
    newLayer->output_size = output_size;
    newLayer->activation = activation;

    if(have_weight){ 
        newLayer->weights = malloc(output_size*input_size*sizeof(float));
        if(!newLayer->weights) goto fail;
        initialize_weight(newLayer->weights,output_size*input_size);
    }
    else{
        newLayer->weights = NULL;
    }
    if (have_bias){
        newLayer->biases = malloc(output_size*sizeof(float));
        if(!newLayer->biases) goto fail;
        initialize_biases(newLayer->biases, output_size);
    }
    else{
        newLayer->biases = NULL;
    }
    newLayer->input_cache = malloc(sizeof(float) * newLayer->input_size);
        if (!newLayer->input_cache) goto fail;    

    newLayer->output_cache = malloc(sizeof(float) * newLayer->output_size);
        if (!newLayer->output_cache) goto fail;

    newLayer->weight_grads = malloc(sizeof(float) * newLayer->input_size * newLayer->output_size);
        if (!newLayer->weight_grads) goto fail;

    newLayer->bias_grads = malloc(sizeof(float) * newLayer->output_size);
        if (!newLayer->bias_grads) goto fail;

    newLayer->delta = malloc(sizeof(float) * newLayer->output_size);
        if (!newLayer->delta) goto fail;

    printLayerInfo(newLayer);

    return 1;

    fail:
        if (newLayer->weights) free(newLayer->weights);
        if (newLayer->biases) free(newLayer->biases);
        if (newLayer->input_cache) free(newLayer->input_cache);
        if (newLayer->output_cache) free(newLayer->output_cache);
        if (newLayer->weight_grads) free(newLayer->weight_grads);
        if (newLayer->bias_grads) free(newLayer->bias_grads);
        if(newLayer->delta) free(newLayer->delta);
    
    return 0;
}



void forwardPropagation(Layer** layers,          // Layers of the network
    int num_layers){
        
    for (int i = 1; i < num_layers; i++){
        Layer* previousLayer = layers[i-1];
        Layer* currentLayer = layers[i];

        float (*activation_func)(float) = get_activation_function(currentLayer->activation);

        memcpy(currentLayer->input_cache, previousLayer->output_cache, sizeof(float) * currentLayer->input_size);

        for (int i = 0; i < currentLayer->output_size; i++) {
            float weighted_sum = 0;
            for (int j = 0; j < currentLayer->input_size; j++) {
                weighted_sum += currentLayer->input_cache[j] * currentLayer->weights[i * currentLayer->input_size + j];
            }

            // Add bias
            weighted_sum += currentLayer->biases[i];

            // Apply activation function and store in output_cache
            currentLayer->output_cache[i] = activation_func(weighted_sum);
        }
    }
}


void backpropagate(
    Layer** layers,          // Layers of the network
    int num_layers,         // Number of layers in the network
    float* targets          // Target output (for the last layer)
) {
    // 1. Calculate error (delta) for the output layer
    int last_layer_index = num_layers - 1;
    Layer* output_layer = layers[last_layer_index];
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
        Layer* current_layer = layers[l];
        Layer* next_layer = layers[l + 1];

        float (*activation_derivative)(float) = get_derivated_activation_function(current_layer->activation);

        
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
                delta += next_layer->weights[j * next_layer->input_size + i] * next_layer->delta[j];
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



void free_layer(Layer* Layer){
    free(Layer->bias_grads);
    free(Layer->biases);
    free(Layer->delta);
    free(Layer->input_cache);
    free(Layer->output_cache);
    free(Layer->weight_grads);
    free(Layer->weights);
    free(Layer);
}


void printLayerInfo(Layer* layer) {
    // Print basic information about the layer
    printf("Layer Information:\n");
    printf("Input Size: %d\n", layer->input_size);
    printf("Output Size: %d\n", layer->output_size);
    
    // Print activation function type
    char* activationstr;
    switch (layer->activation) {
        case RELU:
            activationstr = "ReLU";
            break;
        case SIGMOID:
            activationstr = "Sigmoid";
            break;
        case TANH:
            activationstr = "Tanh";
            break;
        case LINEAR:
            activationstr = "Linear";
            break;
        default:
            activationstr = "Unknown";
    }
    printf("Activation Type: %s\n", activationstr);
    
    // Print weights (only a portion for readability)
    if(layer->weights == NULL){
        printf("last weights: NULL\n");
    }
    else{
        printf("last weights: %f\n", layer->weights[layer->input_size * layer->output_size -1]);
    }
    // Print biases
    if (layer->biases == NULL){
        printf("Biases:NULL\n");
    }
    else{
        printf("Biases:%f\n", layer->biases[layer->output_size - 1]);
    }
    
   
}

