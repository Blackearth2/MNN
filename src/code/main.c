#include "layer.h"
#include "network.h"
#include "image/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define NUM_SAMPLES 10      // Number of training samples
#define IMAGE_SIZE 1024     // 32x32 image flattened into 1024 elements
#define NUM_CLASSES 9      // Number of classes (letters, for example)

void generateRandomData(float** trainingData, float** targets, int numSamples, int inputSize, int numClasses) {
    for (int i = 0; i < numSamples; i++) {
        // Generate random image data (flattened 32x32)
        for (int j = 0; j < inputSize; j++) {
            trainingData[i][j] = (float)rand() / RAND_MAX;  // Random float between 0 and 1
        }

        // Generate valid one-hot encoded target vector
        int targetClass = rand() % numClasses;  // Random class (index for one-hot)
        for (int k = 0; k < numClasses; k++) {
            targets[i][k] = (k == targetClass) ? 1.0f : 0.0f;  // Only one class is 1.0, others are 0.0
        }
    }
}


int main(){

    network* Network = malloc(sizeof(network));
    int nbLayers = 3;
    Network->layers = malloc(nbLayers*sizeof(Layer));
    Network->nbLayers = nbLayers;
    ActivationType activation[] = {RELU,RELU,SIGMOID};
    int inputs_size[] = {0,784,512};
    int output_size[] = {784,512,9};


    for (int i = 0; i < nbLayers; i++)
    {
        int have_bias = i == 0 ? 0 : 1;
        int have_weights = i == 0 ? 0 : 1;

        Network->layers[i] = malloc(sizeof(Layer));
        initialize_layer(Network->layers[i],inputs_size[i],output_size[i],have_bias,have_weights,activation[i]);
    }
    printNetworkInfo(Network);
    printf("\n");
    srand(time(NULL));  // Seed the random number generator

    // Allocate memory for training data and targets
    float** trainingData = (float**)malloc(NUM_SAMPLES * sizeof(float*));
    float** targets = (float**)malloc(NUM_SAMPLES * sizeof(float*));

    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        trainingData[i] = (float*)malloc(IMAGE_SIZE * sizeof(float));  // 32x32 image
        targets[i] = (float*)malloc(NUM_CLASSES * sizeof(float));      // 26-class one-hot target
        if(!targets){
            printf("error during malloc\n");
        }
    }

    // Generate random data
    //generateRandomData(trainingData, targets, NUM_SAMPLES, IMAGE_SIZE, NUM_CLASSES);
    loadFile("/home/amaury/projectPerso/MNN/src/resources/train.csv",trainingData,targets,inputs_size[0],NUM_SAMPLES);

    /*// Print a sample of the data for verification
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("Sample %d:\n", i + 1);
        
        printf("Image: ");
        for (int j = 0; j < 10; j++) {  // Print only the first 10 pixels for brevity
            printf("%.2f ", trainingData[i][j]);
        }
        printf("\n");

        printf("Target: ");
        for (int k = 0; k < NUM_CLASSES; k++) {
            printf("%.1f ", targets[i][k]);
        }
        printf("\n\n");
    }*/



    trainNetwork(Network,1000,trainingData,targets,NUM_SAMPLES);


    freeNetwork(Network);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        free(trainingData[i]);
        free(targets[i]);
    }
    free(trainingData);
    free(targets);


    return 0;
}