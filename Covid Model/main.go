package main // This will need to change and become abstracted

import (
	"fmt"
	// "math"
	// "math/rand"
	"os"
	"gonet"
	"io"
	"strconv"
	// "log"
	// "gonum.org/v1/gonum/mat"
	"encoding/csv"
	"bufio"
	// "image/png"
	// "image"
	
)

func trainCovid(n *gonet.GoNetwork, numEpochs int, path string) {

	for eachEpoch := 0; eachEpoch < numEpochs; eachEpoch++ {
		fmt.Println("Current Epoch: ", eachEpoch)

		// open file
		file, _ := os.Open(path)
		parsed := csv.NewReader(bufio.NewReader(file))
		count := -1
		for {
			count++
			read, err := parsed.Read()
			if err == io.EOF {
				break
			}
			if count == 0 {
				continue
			}
			

			inputData := make([]float64, n.Inputs)

			for i := 1; i < len(read); i++ {
				value, err := strconv.ParseFloat(read[i], 64)
				if err != nil {
					fmt.Println("Could not parse csv values")
				}
				inputData[i-1] = value
			}

			outputData := make([]float64, n.Outputs)

			value, _ := strconv.Atoi(read[0])
			if value == 1 {
				outputData[0] = 0.01
				outputData[1] = 0.99
			} else {
				outputData[0] = 0.99
				outputData[1] = 0.01
			}

			n.TrainFull(inputData, outputData)
		}
		file.Close()
	}
	fmt.Println("Finished Training")
}	

func testCovid(n *gonet.GoNetwork, path string) {

	// open file
	file, _ := os.Open(path)
	defer file.Close()
	parsed := csv.NewReader(bufio.NewReader(file))

	numSuccess, truePos, falsePos, trueNeg, falseNeg := 0, 0.0, 0.0, 0.0, 0.0
	count := -1
	for {
		count++
		read, err := parsed.Read()
		if err == io.EOF {
			break
		} 
		if count == 0 {
			continue
		}
		
		inputData := make([]float64, n.Inputs)

		for i := 1; i < len(read); i++ {
			value, err := strconv.ParseFloat(read[i], 64)
			if err != nil {
				fmt.Println("Could not parse csv values")
			}
			inputData[i-1] = value
		}
		_, _, outputResults := n.ForwardPropagate(inputData)

		value, _ := strconv.Atoi(read[0])
		if outputResults.At(1, 0) > outputResults.At(0, 0) {
			if value == 1 {
				truePos++
				numSuccess++
			} else {
				falsePos++
			}
		} else {
			if value == 1 {
				falseNeg++
			} else {
				trueNeg++
				numSuccess++
			}
		}
	}
	gonet.Accuracy(numSuccess, count)
	gonet.F1Score(truePos, falsePos, falseNeg, trueNeg)
}

func main() {

	// Create the neural net
	covidNet := gonet.MakeGoNetwork(10, 1000, 2, 0.15)

	trainCovid(covidNet, 5, "covid19/dataset_train3.csv")
	// gonet.Upload(covidNet)

	// gonet.Load(covidNet)
	testCovid(covidNet, "covid19/dataset_test3.csv")
}

