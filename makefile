CFLAGS = -Wall -Wextra
POINTS = 2500000

.PHONY: all clean debug re

all: build c run

build: *.h find_events.cu
	nvcc -rdc=true -gencode arch=compute_50,code=compute_50 find_events.cu -o find_events

clean:
	-trash find_events
	-trash transitions_guessed_canny.csv
	-trash transitions_guessed_delta.csv
	-trash transitions_guessed_mean.csv
	-trash signal.csv
	-trash signal.dat
	-trash generated_signal.png

debug:	*.h find_events.cu
	nvcc -rdc=true -gencode arch=compute_50,code=compute_50 --ptxas-options=-v -g -G find_events.cu -o find_events

re:	# thanks, -j4...
	make clean
	make build
	@echo "Cleaned and recompiled..."

signal.csv:
	python signal_generator.py $(POINTS)

signal.dat: signal.csv
	python text_to_binary.py signal.csv
	python plot_data.py one signal.csv

run: build signal.dat # usage here is that the data as argv[2], with the correct option at compile-time
	./find_events mean
	./find_events delta
	./find_events canny

c: build signal.dat
	./find_events c

csv: build signal.csv # usage here is that the data as argv[2], with the correct option at compile-time
	./find_events mean signal.csv
	./find_events delta signal.csv
	./find_events canny signal.csv

plot:
	python plot_csv.py signal.csv
