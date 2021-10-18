# Makefile for l1tf

CC      = gcc
CFLAGS  = -O3
LDFLAGS =

# LIBS    = -L/usr/lib -lblas -llapack -lm
LIBS    = -lblas -llapack -lm

SRCS    = l1tf.c main.c
OBJS    = $(patsubst %.c,%.o,$(SRCS))

PROG    = l1tf

all: $(PROG)

$(OBJS) : %.o : %.c
	$(CC) -c $(CFLAGS) $< -o $@

$(PROG) : $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $(PROG) $(OBJS) $(LIBS)

.PHONY : clean 

clean:
	rm -f *.o *.gcno *.gcda $(PROG)

