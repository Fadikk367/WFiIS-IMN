CXX=gcc
OUTPUT=out
CXXFLAGS=-g -Wall -lm
CXXFLAGS+=-pedantic
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)
all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(CXX) $? -o $@ -lm

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -c $*.c -o $@ -lm

clean:
	rm -f $(OUTPUT) $(OBJ)

run: $(OUTPUT)
	./$(OUTPUT)
