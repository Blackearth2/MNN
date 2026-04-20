#include "network.h"

void initializeNetwork(network* net, int nbLayers,
                       ActivationType* activations,
                       int* input_sizes, int* output_sizes) {
    net->nbLayers    = nbLayers;
    net->input_size  = input_sizes[0];
    net->output_size = output_sizes[nbLayers-1];
    net->layers      = malloc(nbLayers * sizeof(Layer*));
    for (int i = 0; i < nbLayers; i++) {
        net->layers[i] = malloc(sizeof(Layer));
        initialize_layer(net->layers[i], input_sizes[i], output_sizes[i],
                         i>0, i>0, activations[i]);
    }
}

static void shuffle(int* arr, int n) {
    for (int i=n-1; i>0; i--) {
        int j=rand()%(i+1), t=arr[i]; arr[i]=arr[j]; arr[j]=t;
    }
}

int predict_class(network* net, float* input) {
    memcpy(net->layers[0]->output_cache, input,
           net->layers[0]->output_size * sizeof(float));
    forwardPropagation(net->layers, net->nbLayers);
    Layer* out = net->layers[net->nbLayers-1];
    int best=0;
    for (int i=1; i<out->output_size; i++)
        if (out->output_cache[i] > out->output_cache[best]) best=i;
    return best;
}

float eval_accuracy(network* net, float** data, float** targets, int n) {
    int correct=0;
    Layer* out = net->layers[net->nbLayers-1];
    for (int s=0; s<n; s++) {
        int pred  = predict_class(net, data[s]);
        int label = 0;
        for (int i=1; i<out->output_size; i++)
            if (targets[s][i] > targets[s][label]) label=i;
        if (pred==label) correct++;
    }
    return (float)correct/n;
}

/* ─── Weight save ────────────────────────────────────────────────────────────
   Format v2:
     4 bytes  magic "NNW2"
     int32    nbLayers
     int32[]  input_size  per layer  (nbLayers values)
     int32[]  output_size per layer  (nbLayers values)
     int32[]  activation  per layer  (nbLayers values)
     for l=1..nbLayers-1:
       float32[input_size[l] * output_size[l]]   weights (row-major)
       float32[output_size[l]]                   biases
──────────────────────────────────────────────────────────────────────────── */
void saveWeights(network* net, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot save to %s\n", path); return; }

    fwrite("NNW2", 1, 4, f);
    fwrite(&net->nbLayers, sizeof(int), 1, f);

    /* input_sizes */
    for (int i=0; i<net->nbLayers; i++)
        fwrite(&net->layers[i]->input_size,  sizeof(int), 1, f);
    /* output_sizes */
    for (int i=0; i<net->nbLayers; i++)
        fwrite(&net->layers[i]->output_size, sizeof(int), 1, f);
    /* activations */
    for (int i=0; i<net->nbLayers; i++) {
        int act = (int)net->layers[i]->activation;
        fwrite(&act, sizeof(int), 1, f);
    }
    /* weights + biases (skip input layer which has none) */
    for (int i=1; i<net->nbLayers; i++) {
        Layer* l = net->layers[i];
        fprintf(stderr, "  saving layer %d: %d×%d weights\n",
                i, l->output_size, l->input_size);
        fwrite(l->weights, sizeof(float), (size_t)l->input_size * l->output_size, f);
        fwrite(l->biases,  sizeof(float), l->output_size, f);
    }
    fclose(f);
    printf("{\"type\":\"saved\",\"path\":\"%s\"}\n", path);
    fflush(stdout);
}

/* ─── Training ─────────────────────────────────────────────────────────────── */
void trainNetwork(network* net, int epochs,
                  float** trainingData, float** targets,
                  int numSamples, int report_every,
                  const char* save_path) {

    Layer* out      = net->layers[net->nbLayers-1];
    int    out_size = out->output_size;

    /* Emit config */
    printf("{\"type\":\"config\",\"layers\":[");
    for (int i=0; i<net->nbLayers; i++) {
        printf("%d", net->layers[i]->output_size);
        if (i < net->nbLayers-1) printf(",");
    }
    printf("],\"epochs\":%d,\"samples\":%d}\n", epochs, numSamples);
    fflush(stdout);

    int* idx = malloc(numSamples * sizeof(int));
    for (int i=0; i<numSamples; i++) idx[i]=i;

    for (int ep=0; ep<epochs; ep++) {
        shuffle(idx, numSamples);
        float total_loss=0;

        for (int s=0; s<numSamples; s++) {
            float* input  = trainingData[idx[s]];
            float* target = targets[idx[s]];

            memcpy(net->layers[0]->output_cache, input,
                   net->layers[0]->output_size * sizeof(float));
            forwardPropagation(net->layers, net->nbLayers);

            total_loss += compute_cross_entropy_loss(out->output_cache, target, out_size);
            backpropagate(net->layers, net->nbLayers, target);

            for (int l=1; l<net->nbLayers; l++) {
                Layer* cl = net->layers[l];
                stochasticGradientDescent(cl->weights, cl->weight_grads,
                                          cl->biases,  cl->bias_grads,
                                          cl->input_size, cl->output_size);
            }

            if (report_every>0 && (s+1)%report_every==0) {
                printf("{\"type\":\"progress\",\"epoch\":%d,\"step\":%d,\"total\":%d,\"loss\":%.6f}\n",
                       ep+1, s+1, numSamples, total_loss/(s+1));
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / numSamples;
        int   eval_n   = numSamples < 5000 ? numSamples : 5000;
        float acc      = eval_accuracy(net, trainingData, targets, eval_n);

        printf("{\"type\":\"epoch\",\"epoch\":%d,\"total_epochs\":%d,\"loss\":%.6f,\"accuracy\":%.4f}\n",
               ep+1, epochs, avg_loss, acc);
        fflush(stdout);
    }

    if (save_path && save_path[0]) saveWeights(net, save_path);

    printf("{\"type\":\"done\"}\n");
    fflush(stdout);
    free(idx);
}

void freeNetwork(network* net) {
    for (int i=0; i<net->nbLayers; i++) free_layer(net->layers[i]);
    free(net->layers);
    free(net);
}
