CC = gcc
CFLAGS = -Wall -Wextra -Werror -g -fsanitize=address -Ineural_network -Iimage
LDFLAGS = -lm
TARGET = nn_model

# Object files with proper paths
OBJECTS = main.o neural_network/layer.o neural_network/network.o image/image.o

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

main.o: main.c neural_network/layer.h neural_network/network.h
	$(CC) $(CFLAGS) -c main.c -o main.o

neural_network/layer.o: neural_network/layer.c neural_network/layer.h
	$(CC) $(CFLAGS) -c neural_network/layer.c -o neural_network/layer.o

neural_network/network.o: neural_network/network.c neural_network/network.h
	$(CC) $(CFLAGS) -c neural_network/network.c -o neural_network/network.o

image/image.o: image/image.c image/image.h
	$(CC) $(CFLAGS) -c image/image.c -o image/image.o
	
clean:
	rm -f $(TARGET) main.o neural_network/*.o