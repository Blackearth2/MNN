#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct{
    Layer** layers;
    int nbLayers;
}network;


void initializeNetwork(network* network,int nbLayers, ActivationType* activations, int* input_size,int* output_size);
void printNetworkInfo(network* network);
void freeNetwork(network* network);

void trainNetwork(  network* net,                     //network to train
                    int epochs,                      //number of epochs to do  
                    float** trainingData,           // trainingData[numSamples][1024] images on each line  
                    float** targets,               // targets[numSamples][26] just lower letter at first
                    int numSamples                //nbData
);

#endif