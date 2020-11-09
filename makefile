CFLAGS = -Wall -Wextra

POINTS = 250000

NAME = find_events
MIN_SRC = find_events.cu
SRC_CU = find_events.cu
GEN_PY = signal_generator.py

.PHONY: all clean debug re

all: *.h $(SRC_CU)
	nvcc -rdc=true -gencode arch=compute_50,code=compute_50 $(SRC_CU) -o $(NAME)

clean:
	-trash $(NAME)
	-trash transitions_guessed_canny.csv
	-trash transitions_guessed_delta.csv
	-trash transitions_guessed_mean.csv
	-trash signal.csv
	-trash signal.dat
	-trash generated_signal.png

debug:	*.h $(SRC_CU)
	nvcc -rdc=true -gencode arch=compute_50,code=compute_50 --ptxas-options=-v -g -G $(SRC_CU) -o $(NAME)

re:	# thanks, -j4...
	make clean
	make all
	@echo "Cleaned and recompiled..."

signal.csv:
	python $(GEN_PY) $(POINTS)

signal.dat: signal.csv
	python text_to_binary.py signal.csv
	python plot_data.py one signal.csv

run: all # usage here is that the data as argv[2], with the correct option at compile-time
	./$(NAME) mean
	./$(NAME) delta
	./$(NAME) canny

csv: all # usage here is that the data as argv[2], with the correct option at compile-time
	./$(NAME) mean signal.csv
	./$(NAME) delta signal.csv
	./$(NAME) canny signal.csv

plot:
	python plot_csv.py signal.csv

