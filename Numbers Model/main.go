package main

import (
	"fmt"
	"os"
	"io"
	"strconv"
	"encoding/csv"
	"bufio"
	"image/png"
	"image"
	"gonet"
)

func trainNumbers(n *gonet.GoNetwork) {
	for epochs := 0; epochs < 1; epochs++ {

		fmt.Println("Currently on epoch # ", epochs)

		csvFile, _  := os.Open("mnist_train_short.csv")
		parse := csv.NewReader(bufio.NewReader(csvFile))
		for {
			record, err := parse.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, n.Inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			n.TrainFull(inputs, targets)
		}
		csvFile.Close()
	}
}

func testNumbers(n *gonet.GoNetwork) {

	checkFile, _ := os.Open("mnist_test_short.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, n.Inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		_, _, outputs := n.ForwardPropagate(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < n.Outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}
	fmt.Print("The score we got is: ", score)
}

func dataFromImage(filePath string) (pixels []float64) {
	
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot decode file: ", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file: ", err)
	}
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	pixels = make([]float64, len(gray.Pix))
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}
	return
}

func predictFromImage(n *gonet.GoNetwork, path string) int {
	input := dataFromImage(path)
	_, _, output := n.ForwardPropagate(input)
	fmt.Print(output)
	best := 0 
	highest := 0.0
	for i := 0; i < n.Outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}

func main() {
	// Create the neural net
	numbersNet := gonet.MakeGoNetwork(784, 200, 10, 1)

	trainNumbers(numbersNet)
	gonet.Upload(numbersNet)

	gonet.Load(numbersNet)
	testNumbers(numbersNet)

	fmt.Println("The prediction is: ", predictFromImage(numbersNet, "numbers_png/image3.png"))
}
