CFLAGS=-Wall
LDFLAGS=-lgsl -ltensor -lcblas -latlas -lm -L/usr/local/atlas/lib
OUTPUT=VanVleck
CC = gcc

all: $(OUTPUT)

VanVleck: SchwVanVleck.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(OUTPUT) SchwVanVleck.c
	
clean:
	rm -f VanVleck
  