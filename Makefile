# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60 -std=c++14 -I$(SRCDIR) -DSINGLE

# Precision options
# -DSINGLE
# -DDOUBLE

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = .

# Source and object files
CUSOURCES = $(wildcard $(SRCDIR)/*/*.cu)
CUOBJECTS = $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(CUSOURCES:.cu=.o))

all: directories run

# Create the necessary directories
directories:
	@mkdir -p $(OBJDIR)

run: $(CUOBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $(BINDIR)/run

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -rf $(OBJDIR)/*
	rm -f $(BINDIR)/run
