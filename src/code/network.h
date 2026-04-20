#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    Layer** layers;
    int     nbLayers;
    int     input_size;
    int     output_size;
} network;

void  initializeNetwork(network* net, int nbLayers,
                        ActivationType* activations,
                        int* input_sizes, int* output_sizes);

void  trainNetwork(network* net, int epochs,
                   float** trainingData, float** targets,
                   int numSamples, int report_every,
                   const char* save_path);

void  saveWeights(network* net, const char* path);
void  freeNetwork(network* net);
int   predict_class(network* net, float* input);
float eval_accuracy(network* net, float** data, float** targets, int n);

#endif
