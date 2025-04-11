#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* labelToTargets(char label,int nbLabels);
void loadFile(char* filepath, float** trainingData, float** targets, int input_size, int num_samples);
void printDataSample(float** targets);

#endif