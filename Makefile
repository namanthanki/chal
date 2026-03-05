VERSION = 1.1
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -pedantic -std=gnu90 -DVERSION=\"$(VERSION)\"

# Cross-platform command definitions
ifeq ($(OS),Windows_NT)
    TARGET = bin\chal.exe
    RM     = del /Q /F
    MKDIR  = if not exist bin mkdir bin
else
    TARGET = bin/chal
    RM     = rm -f
    MKDIR  = mkdir -p bin
endif

all: $(TARGET)

# Compile the single file directly to the output folder
$(TARGET): src/chal.c
	$(MKDIR)
	$(CC) $(CFLAGS) src/chal.c -o $(TARGET)

clean:
	$(RM) $(TARGET)
