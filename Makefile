CFLAGS=-Wall -I/opt/local/include
LDFLAGS=-L/usr/local/atlas/lib -L/opt/local/lib -lgsl -lcblas -latlas -lm
CC = gcc

all: VanVleck Geodesics

VanVleck: SchwVanVleck.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ SchwVanVleck.c

Geodesics: SchwGeodesicEqns.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ SchwGeodesicEqns.c
	
clean:
	rm -f VanVleck Geodesics
  