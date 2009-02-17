CFLAGS=-Wall -I/opt/local/include
LDFLAGS=-L/usr/local/atlas/lib -L/opt/local/lib -lgsl -lcblas -latlas -lm
CC = gcc

all: VanVleck Geodesics

VanVleckNariai: VanVleck.c NariaiTensors.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ VanVleck.c NariaiTensors.c

VanVleckSchw: VanVleck.c SchwTensors.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ VanVleck.c SchwTensors.c

Geodesics: SchwGeodesicEqns.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ SchwGeodesicEqns.c
	
clean:
	rm -f VanVleck Geodesics
  