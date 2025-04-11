
#include "image.h"



void loadFile(char* filepath, float** trainingData, float** targets,int input_size, int num_samples){
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        printf("Erreur : Impossible d'ouvrir le fichier %s\n", filepath);
    }

    char line[4000];
    int row = 0;

    //skip first line (header)


    while(fgets(line,4000,fp) && row < num_samples){
        printf("%.20s\n", line);
        char *token = strtok(line, ",");
        targets[row] = labelToTargets(token[0], 10);

        token = strtok(NULL, ",");

        for (int col = 0; col < input_size; col++) {
            trainingData[row][col] = atof(token) / 250.0f ;
            token = strtok(NULL, ",");
        }
        row++;
    }
    printDataSample(targets);
    fclose(fp);
}




float* labelToTargets(char label,int nbLabels){
    //printf("%c",label);
    float labelToFloat = atof(&label);
    //printf(", %f\n",labelToFloat);
    float* newLabelTarget = malloc(nbLabels*sizeof(float));
    for (int i = 0; i < nbLabels; i++){
        if (labelToFloat == (float)i){
            newLabelTarget[i] = 1.0f;
        }
        else {
            newLabelTarget[i] = 0.0f;
        }
    }
    return newLabelTarget;
}


void printDataSample(float** targets){
    for (int i = 0; i < 10; i++)
    {
        printf("Label %d:\n",i);
        for(int j = 0; j < 10;j++){
            printf("%f,",targets[i][j]);
            printf("\n");
        }
    }
    
}