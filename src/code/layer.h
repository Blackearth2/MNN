#ifndef LAYER_H
#define LAYER_H

#include <err.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LEARNING_RATE 0.01f

extern float g_learning_rate;

typedef enum { RELU, SIGMOID, TANH, LINEAR, SOFTMAX } ActivationType;

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
    ActivationType activation;
    float *input_cache;
    float *output_cache;
    float *weight_grads;
    float *bias_grads;
    float *delta;
} Layer;

int   initialize_layer(Layer* L, int input_size, int output_size,
                        int have_bias, int have_weight, ActivationType activation);
void  forwardPropagation(Layer** layers, int num_layers);
void  backpropagate(Layer** layers, int num_layers, float* targets);
void  softmax_inplace(float* v, int n);

float relu(float x);
float sigmoid(float x);
float tanh_func(float x);
float relu_derivative(float x);

float compute_mse_loss(float* pred, float* target, int size);
float compute_cross_entropy_loss(float* pred, float* target, int size);

void batchGradientDescent(float* weights, float* wg, float* biases, float* bg,
                           int input_size, int output_size, int m);
void stochasticGradientDescent(float* weights, float* wg, float* biases, float* bg,
                                int input_size, int output_size);
void miniBatchGradientDescent(float* weights, float* wg, float* biases, float* bg,
                               int input_size, int output_size, int mini_batch_size);

void printLayerInfo(Layer* L);
void free_layer(Layer* L);

#endif
