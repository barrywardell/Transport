CFLAGS=-Wall
LDFLAGS=-L/usr/local/atlas/lib -lgsl -lcblas -latlas -lm
OUTPUT=VanVleck
CC = gcc

all: $(OUTPUT)

VanVleck: SchwVanVleck.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(OUTPUT) SchwVanVleck.c
	
clean:
	rm -f VanVleck
  