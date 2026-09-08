VERSION ?= 2.0.0
CC      = gcc
ARCH    ?= native

# Base compilation flags
BASE_FLAGS = -O3 -Wall -Wextra -pedantic -std=gnu99 -DVERSION=\"$(VERSION)\"

# Architecture-specific flags
ifeq ($(ARCH),avx2)
    ARCH_FLAGS = -mavx2 -mbmi2 -mpopcnt
else ifeq ($(ARCH),general)
    ARCH_FLAGS = -msse4.2 -mpopcnt
else ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=$(ARCH)
endif

CFLAGS  ?= $(BASE_FLAGS) $(ARCH_FLAGS)
LDFLAGS = -lm

# Cross-platform extensions & commands
ifeq ($(OS),Windows_NT)
    EXT   = .exe
    RM    = powershell -Command "Remove-Item -Path bin/chal* -ErrorAction SilentlyContinue"
    MKDIR = powershell -Command "if(-not(Test-Path bin)){New-Item -ItemType Directory -Path bin}"
else
    EXT   =
    RM    = rm -f bin/chal*
    MKDIR = mkdir -p bin
endif

TARGET ?= bin/chal$(EXT)

.PHONY: all native avx2 general debug bench perft clean

all: $(TARGET)

native:
	$(MAKE) ARCH=native TARGET=bin/chal$(EXT)

avx2:
	$(MAKE) ARCH=avx2 TARGET=bin/chal-avx2$(EXT)

general:
	$(MAKE) ARCH=general TARGET=bin/chal-general$(EXT)

debug:
	$(MAKE) CFLAGS="-g -O0 -Wall -Wextra -pedantic -std=gnu99 -DVERSION=\"$(VERSION)\"" TARGET=bin/chal-debug$(EXT)

$(TARGET): src/chal.c
	$(MKDIR)
	$(CC) $(CFLAGS) src/chal.c -o $(TARGET) $(LDFLAGS)

bench: $(TARGET)
	$(TARGET) bench

perft: $(TARGET)
	$(TARGET) perft 5

clean:
	$(RM)
