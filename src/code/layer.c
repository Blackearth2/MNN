#include "layer.h"

float g_learning_rate = LEARNING_RATE;

static void validateSize(int in, int out) {
    if (in <= 0 || out <= 0) err(1, "layer sizes must be positive");
}

/* ── Activations ── */
float relu(float x)      { return x > 0 ? x : 0; }
float sigmoid(float x)   { return 1.0f / (1.0f + expf(-x)); }
float tanh_func(float x) { return tanhf(x); }
static float linear_f(float x) { return x; }

float relu_derivative(float x)          { return x > 0 ? 1.0f : 0.0f; }
static float sigmoid_derivative(float x){ float s = sigmoid(x); return s*(1-s); }
static float tanh_derivative(float x)   { float t = tanhf(x); return 1-t*t; }
static float linear_derivative(float x) { (void)x; return 1.0f; }

void softmax_inplace(float* v, int n) {
    float mx = v[0];
    for (int i=1;i<n;i++) if(v[i]>mx) mx=v[i];
    float sum=0;
    for (int i=0;i<n;i++) { v[i]=expf(v[i]-mx); sum+=v[i]; }
    for (int i=0;i<n;i++) v[i]/=sum;
}

static float (*get_act(ActivationType t))(float) {
    switch(t){
        case RELU: return relu; case SIGMOID: return sigmoid;
        case TANH: return tanh_func; case LINEAR: case SOFTMAX: return linear_f;
        default: err(1,"bad activation");
    }
}
static float (*get_dact(ActivationType t))(float) {
    switch(t){
        case RELU: return relu_derivative; case SIGMOID: return sigmoid_derivative;
        case TANH: return tanh_derivative; case LINEAR: case SOFTMAX: return linear_derivative;
        default: err(1,"bad activation");
    }
}

/* ── Loss ── */
float compute_mse_loss(float* p, float* t, int n){
    float s=0; for(int i=0;i<n;i++){float e=p[i]-t[i];s+=e*e;} return s/n;
}
float compute_cross_entropy_loss(float* p, float* t, int n){
    float s=0; for(int i=0;i<n;i++) s+=t[i]*logf(p[i]+1e-15f); return -s;
}

/* ── Gradient Descent ── */
void batchGradientDescent(float* w,float* wg,float* b,float* bg,int in,int out,int m){
    for(int i=0;i<out;i++){
        for(int j=0;j<in;j++) w[i*in+j]-=g_learning_rate*wg[i*in+j]/m;
        b[i]-=g_learning_rate*bg[i]/m;
    }
}
void stochasticGradientDescent(float* w,float* wg,float* b,float* bg,int in,int out){
    for(int i=0;i<out;i++){
        for(int j=0;j<in;j++) w[i*in+j]-=g_learning_rate*wg[i*in+j];
        b[i]-=g_learning_rate*bg[i];
    }
}
void miniBatchGradientDescent(float* w,float* wg,float* b,float* bg,int in,int out,int mb){
    for(int i=0;i<out;i++){
        for(int j=0;j<in;j++) w[i*in+j]-=g_learning_rate*wg[i*in+j]/mb;
        b[i]-=g_learning_rate*bg[i]/mb;
    }
}

/* ── Weight init (Xavier) ── */
static void init_weights(float* w, int in, int out){
    float lim=sqrtf(6.0f/(in+out));
    for(int i=0;i<in*out;i++) w[i]=lim*(2.0f*rand()/RAND_MAX-1.0f);
}

/* ── Layer init ── */
int initialize_layer(Layer* L,int in,int out,int hb,int hw,ActivationType act){
    validateSize(in,out);
    L->input_size=in; L->output_size=out; L->activation=act;
    L->weights=L->biases=L->input_cache=L->output_cache=
    L->weight_grads=L->bias_grads=L->delta=NULL;

    if(hw){ L->weights=malloc(in*out*sizeof(float)); if(!L->weights) goto fail; init_weights(L->weights,in,out); }
    if(hb){ L->biases=calloc(out,sizeof(float));     if(!L->biases)  goto fail; }

    L->input_cache  = calloc(in,      sizeof(float)); if(!L->input_cache)  goto fail;
    L->output_cache = calloc(out,     sizeof(float)); if(!L->output_cache) goto fail;
    L->weight_grads = calloc(in*out,  sizeof(float)); if(!L->weight_grads) goto fail;
    L->bias_grads   = calloc(out,     sizeof(float)); if(!L->bias_grads)   goto fail;
    L->delta        = calloc(out,     sizeof(float)); if(!L->delta)        goto fail;
    return 1;
fail:
    free(L->weights);free(L->biases);free(L->input_cache);
    free(L->output_cache);free(L->weight_grads);free(L->bias_grads);free(L->delta);
    return 0;
}

/* ── Forward ── */
void forwardPropagation(Layer** layers, int num_layers){
    for(int i=1;i<num_layers;i++){
        Layer* p=layers[i-1]; Layer* c=layers[i];
        float(*act)(float)=get_act(c->activation);
        memcpy(c->input_cache,p->output_cache,c->input_size*sizeof(float));
        for(int n=0;n<c->output_size;n++){
            float s=c->biases?c->biases[n]:0;
            for(int j=0;j<c->input_size;j++) s+=c->input_cache[j]*c->weights[n*c->input_size+j];
            c->output_cache[n]=act(s);
        }
        if(c->activation==SOFTMAX) softmax_inplace(c->output_cache,c->output_size);
    }
}

/* ── Backprop ── */
void backpropagate(Layer** layers,int num_layers,float* targets){
    /* Output layer */
    Layer* out=layers[num_layers-1];
    float(*dact)(float)=get_dact(out->activation);
    for(int i=0;i<out->output_size;i++){
        float e=out->output_cache[i]-targets[i];
        out->delta[i]      = (out->activation==SOFTMAX)?e : e*dact(out->output_cache[i]);
        out->bias_grads[i] = out->delta[i];
        for(int j=0;j<out->input_size;j++)
            out->weight_grads[i*out->input_size+j]=out->delta[i]*out->input_cache[j];
    }
    /* Hidden layers */
    for(int l=num_layers-2;l>=1;l--){
        Layer* c=layers[l]; Layer* nx=layers[l+1];
        float(*dact2)(float)=get_dact(c->activation);
        for(int i=0;i<c->output_size;i++){
            float d=0;
            for(int j=0;j<nx->output_size;j++) d+=nx->weights[j*nx->input_size+i]*nx->delta[j];
            c->delta[i]      = d*dact2(c->output_cache[i]);
            c->bias_grads[i] = c->delta[i];
            for(int j=0;j<c->input_size;j++)
                c->weight_grads[i*c->input_size+j]=c->delta[i]*c->input_cache[j];
        }
    }
}

/* ── Utils ── */
void free_layer(Layer* L){
    free(L->weights);free(L->biases);free(L->input_cache);
    free(L->output_cache);free(L->weight_grads);free(L->bias_grads);
    free(L->delta);free(L);
}
void printLayerInfo(Layer* L){
    const char* n[]={"RELU","SIGMOID","TANH","LINEAR","SOFTMAX"};
    fprintf(stderr,"  [%d→%d] %s w=%s b=%s\n",
        L->input_size,L->output_size,n[L->activation],
        L->weights?"yes":"none",L->biases?"yes":"none");
}
