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

func parseData(n *gonet.GoNetwork, numEpochs int, path string) {

	for eachEpoch := 0; eachEpoch < numEpochs; eachEpoch++ {
		fmt.Println("Current Epoch: ", eachEpoch)

		// open file
		file, _ := os.Open(path)
		parsed := csv.NewReader(bufio.NewReader(file))
		count := 0
		for {
			read, err := parsed.Read()
			if err == io.EOF {
				break
			}
			if count == 0 {
				continue
			}

			inputData := make([]float64, n.Inputs)

			for i := 1; i < len(inputData); i++ {
				value, err := strconv.ParseFloat(read[i], 64)
				if err != nil {
					fmt.Println("Could not parse csv values")
				}
				inputData[i] = value
			}

			outputData := make([]float64, n.Outputs)

			value, _ := strconv.ParseInt(read[0], 10, 64)
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

func main() {
	fmt.Println("Hello humans")

	// Create the neural net
	numbersNet := gonet.MakeGoNetwork(10, 10, 4, 0.1)

	parseData(numbersNet, 2, "covid19/dataset_clean_2.csv")

	
}

