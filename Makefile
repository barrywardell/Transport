CFLAGS=-Wall -g -O3 -I/opt/local/include
LDFLAGS=-L/opt/local/lib -lgsl -lcblas -latlas -lm
CC = gcc

.PHONY: all V0 VanVleck Geodesics

all: VanVleck Geodesics V0

V0: V0Nariai V0Schw

VanVleck: VanVleckNariai VanVleckSchw

Geodesics: SchwGeodesics

VanVleckNariai: VanVleck.c NariaiTensors.c VanVleckEquations.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ VanVleck.c NariaiTensors.c VanVleckEquations.c

VanVleckSchw: VanVleck.c SchwTensors.c VanVleckEquations.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ VanVleck.c SchwTensors.c VanVleckEquations.c

V0Nariai: V0.c NariaiTensors.c VanVleckEquations.c V0Equations.c
	$(CC) -D NARIAI $(CFLAGS) $(LDFLAGS) -o $@ V0.c NariaiTensors.c VanVleckEquations.c V0Equations.c

V0Schw: V0.c SchwTensors.c VanVleckEquations.c V0Equations.c
	$(CC) -D SCHWARZSCHILD $(CFLAGS) $(LDFLAGS) -o $@ V0.c SchwTensors.c VanVleckEquations.c V0Equations.c

SchwGeodesics: SchwGeodesicEqns.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ SchwGeodesicEqns.c
	
clean:
	rm -f V0Nariai V0Schw VanVleckNariai VanVleckSchw SchwGeodesics

fullclean: clean
	rm -f *.csv

