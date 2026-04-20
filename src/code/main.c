/*
 * main.c
 * Usage:
 *   nn_train <data.bin> <epochs> <lr> <report_every> <model_save_path>
 *            <act0..N> <size0..N>
 *
 * Pass "" as model_save_path to skip saving.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "network.h"

static void die(const char* m) { fprintf(stderr,"ERROR: %s\n",m); exit(1); }

int main(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: nn_train <data.bin> <epochs> <lr> <report_every>"
            " <save_path> <act0..N> <size0..N>\n");
        return 1;
    }
    srand((unsigned)time(NULL));

    const char* data_path  = argv[1];
    int   epochs           = atoi(argv[2]);
    float lr               = (float)atof(argv[3]);
    int   report_every     = atoi(argv[4]);
    const char* save_path  = argv[5];  /* "" to skip */

    int remaining = argc - 6;
    if (remaining % 2 != 0) die("odd number of layer args");
    int num_layers = remaining / 2;

    ActivationType* acts  = malloc(num_layers * sizeof(ActivationType));
    int*            sizes = malloc(num_layers * sizeof(int));
    for (int i=0; i<num_layers; i++) acts[i]  = (ActivationType)atoi(argv[6+i]);
    for (int i=0; i<num_layers; i++) sizes[i] = atoi(argv[6+num_layers+i]);

    /* Load binary data */
    FILE* f = fopen(data_path,"rb");
    if (!f) { perror(data_path); return 1; }

    int num_samples, input_size, output_size;
    if (fread(&num_samples, sizeof(int),1,f) != 1 ||
        fread(&input_size,  sizeof(int),1,f) != 1 ||
        fread(&output_size, sizeof(int),1,f) != 1) die("bad header");

    float* flat_in  = malloc((size_t)num_samples * input_size  * sizeof(float));
    float* flat_lbl = malloc((size_t)num_samples * output_size * sizeof(float));
    if (!flat_in || !flat_lbl) die("out of memory");

    if (fread(flat_in,  sizeof(float), (size_t)num_samples*input_size,  f) != (size_t)num_samples*input_size  ||
        fread(flat_lbl, sizeof(float), (size_t)num_samples*output_size, f) != (size_t)num_samples*output_size)
        die("truncated data file");
    fclose(f);

    float** inputs = malloc(num_samples*sizeof(float*));
    float** labels = malloc(num_samples*sizeof(float*));
    for (int i=0; i<num_samples; i++) {
        inputs[i] = flat_in  + (size_t)i*input_size;
        labels[i] = flat_lbl + (size_t)i*output_size;
    }

    g_learning_rate = lr;

    /*
     * BUG FIX: sizes[] = [input, h1, h2, ..., output]
     * Layer i must have:
     *   input_size  = sizes[i-1]   (previous layer's output count)
     *   output_size = sizes[i]     (this layer's neuron count)
     * Passing sizes,sizes made every layer a SQUARE matrix —
     * the network was blind to all but the first sizes[i] inputs.
     */
    int* in_sizes  = malloc(num_layers * sizeof(int));
    int* out_sizes = malloc(num_layers * sizeof(int));
    in_sizes[0] = out_sizes[0] = sizes[0];   /* input layer: no weights, both = input_size */
    for (int i = 1; i < num_layers; i++) {
        in_sizes[i]  = sizes[i-1];  /* previous layer's neuron count */
        out_sizes[i] = sizes[i];    /* this layer's neuron count     */
    }

    /* Sanity check — print resolved architecture to stderr */
    fprintf(stderr, "Architecture:\n");
    for (int i = 0; i < num_layers; i++)
        fprintf(stderr, "  layer %d: %d → %d (act=%d)\n",
                i, in_sizes[i], out_sizes[i], (int)acts[i]);

    network* net = malloc(sizeof(network));
    initializeNetwork(net, num_layers, acts, in_sizes, out_sizes);

    trainNetwork(net, epochs, inputs, labels, num_samples, report_every, save_path);

    freeNetwork(net);
    free(flat_in); free(flat_lbl);
    free(inputs);  free(labels);
    free(acts); free(sizes);
    free(in_sizes); free(out_sizes);
    return 0;
}
