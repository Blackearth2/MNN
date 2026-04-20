CC     = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LIBS   = -lm

SRC = layer.c network.c main.c
BIN = nn_train

all: $(BIN)

$(BIN): $(SRC) layer.h network.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LIBS)

clean:
	rm -f $(BIN)
