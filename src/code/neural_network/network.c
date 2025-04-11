
#include "network.h"

void initializeNetwork(network* network,int nbLayers, ActivationType* activations, int* input_size,int* output_size){
    network->nbLayers = nbLayers;
    network->layers = malloc(nbLayers*sizeof(Layer));

    for (int i = 0; i < nbLayers; i++)
    {
        int has_bias = i == 0 ? 0 : 1;
        int has_weight = i == 0 ? 0 : 1;
        network->layers[i] = malloc(sizeof(Layer));
        initialize_layer(network->layers[i],input_size[i],output_size[i],has_bias,has_weight,activations[i]);

    }
    
}



void trainNetwork(  network* net,                     //network to train
                    int epochs,                      //number of epochs to do  
                    float** trainingData,           // trainingData[numSamples][1024] images on each line  
                    float** targets,               // targets[numSamples][26] just lower letter at first
                    int numSamples                //nbData
){

    for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++){
        float total_loss = 0.0f;

        printf("Starting Epoch %d\n", currentEpoch + 1);

        for (int currentSample = 0; currentSample < numSamples; currentSample++){
            // get current sample
            float* input = trainingData[currentSample];
            float* target = targets[currentSample];

            /*// Debug: Print current sample and target
            printf("\nSample %d:\n", currentSample + 1);
            printf("Input: ");
            for (int i = 0; i < net->layers[0]->input_size; i++) {
                printf("%f ", input[i]);
            }
            printf("\nTarget: ");
            for (int i = 0; i < 26; i++) {
                printf("%f ", target[i]);
            }
            printf("\n");*/

            // Feed first layer (input layer)
            memcpy(net->layers[0]->output_cache, input, sizeof(float) * net->layers[0]->output_size);

            /*// Debug: Print the output of the input layer
            printf("Layer 0 output: ");
            for (int i = 0; i < net->layers[0]->output_size; i++) {
                printf("%f ", net->layers[0]->output_cache[i]);
            }
            printf("\n");
            */
            // Feedforward
            forwardPropagation(net->layers, net->nbLayers);

            /*
            // Debug: Print all layer outputs
            for (int i = 1; i < net->nbLayers; i++) {
                printf("Layer %d output: ", i);
                for (int j = 0; j < net->layers[i]->output_size; j++) {
                    printf("%f ", net->layers[i]->output_cache[j]);
                }
                printf("\n");
            }*/

            // Check if layers are properly initialized
            if (net->layers == NULL) {
                printf("Error: net->layers is NULL\n");
                return;
            }

            // Check if the number of layers is valid
            if (net->nbLayers <= 0) {
                printf("Error: Invalid number of layers, net->nbLayers = %d\n", net->nbLayers);
                return;
            }

            // Check if the last layer is valid
            if (net->layers[net->nbLayers-1] == NULL) {
                printf("Error: Last layer (layer %d) is NULL\n", net->nbLayers-1);
                return;
            }

            // Check if output_cache of the last layer is properly allocated
            if (net->layers[net->nbLayers-1]->output_cache == NULL) {
                printf("Error: output_cache of the last layer is NULL\n");
                return;
            }

            // Compute loss
            //TODO MAGIC NUMBER a la con
            float loss = compute_mse_loss(net->layers[net->nbLayers-1]->output_cache, target, 9);
            total_loss += loss;
            //printf("Loss for this sample: %f\n", loss);

            // Backpropagation
            backpropagate(net->layers, net->nbLayers, target);

            // Gradient Descent
            for (int i = 1; i < net->nbLayers; i++){
                Layer* currentLayer = net->layers[i];
                batchGradientDescent(currentLayer->weights,
                                     currentLayer->weight_grads,
                                     currentLayer->biases,
                                     currentLayer->bias_grads,
                                     currentLayer->input_size,
                                     currentLayer->output_size,
                                     numSamples);
            }
        }

        printf("Epoch %d - Total Loss: %f\n", currentEpoch + 1, total_loss / numSamples);
    }
}


void printNetworkInfo(network* network){
    printf("beginning printing net info: \n");
    for (int i = 0; i < network->nbLayers;i++){
        printLayerInfo(network->layers[i]);
    }
}

void freeNetwork(network* network){
    for (int i = 0; i < network->nbLayers;i++){
        free_layer(network->layers[i]);
    }
    free(network->layers);
    free(network);
}