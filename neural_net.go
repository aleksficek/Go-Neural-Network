package gonet

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"io"
	"strconv"
	"log"
	"gonum.org/v1/gonum/mat"
	"encoding/csv"
	"bufio"
	"image/png"
	"image"
)

// This is the start of the neural net in golang!

// Node is a data structure that represents a neural network node
type Node struct {
	value int
}

// GoNetwork is the neural network struct
type GoNetwork struct {
	speedOfNetwork float64
	Inputs         int
	Mids           int
	Outputs        int
	firstWeights   *mat.Dense
	secondWeights  *mat.Dense
}

// MakeGoNetwork creates an instance of the neural network
func MakeGoNetwork(inputs, mids, outputs int, speedOfNetwork float64) *GoNetwork{

	neuralNet := GoNetwork{
		speedOfNetwork: speedOfNetwork,
		Inputs:         inputs,
		Mids:           mids,
		Outputs:        outputs,
		firstWeights:   mat.NewDense(mids, inputs, createRandomArray(mids*inputs)),
		secondWeights:  mat.NewDense(outputs, mids, createRandomArray(outputs*mids)),
	}
	fmt.Println("Creating neural net structure: ", neuralNet)
	return &neuralNet
}

// TrainForwards performs a single iteration of forward propagation 
func (n *GoNetwork) TrainForwards(data []float64) (mat.Matrix, mat.Matrix, mat.Matrix) {
	newInputs := mat.NewDense(n.Inputs, 1, data)
	newMids := dotProduct(n.firstWeights, newInputs)
	newMidsActivated := activateMatrix(sigmoid, newMids)
	newOuts := dotProduct(n.secondWeights, newMidsActivated)
	return newInputs, newMidsActivated, activateMatrix(sigmoid, newOuts)
}

// GetError finds the error between expected and forward propagated data
func (n *GoNetwork) GetError(resultData []float64, resultOutputs mat.Matrix) (mat.Matrix, mat.Matrix) {
	result := mat.NewDense(len(resultData), 1, resultData)
	outsError := subtract(result, resultOutputs)
	midsError := dotProduct(n.secondWeights.T(), outsError)
	return midsError, outsError
}

// TrainBackwards performs a single iteration of backwards propagation
func (n *GoNetwork) TrainBackwards(inputs, activatedMids, activatedOuts, midsError, outsError mat.Matrix) {

	// 1. Multiply outsError and sigprime of activated outs
	// 2. Dot of this with activtedmids
	// 3. Scale this by the learning rate

	multipliedMatrix := multiply(outsError, sigInverse(activatedOuts))
	dottedMatrix := dotProduct(multipliedMatrix, activatedMids.T())
	scaledMatrix := scale(dottedMatrix, n.speedOfNetwork)
	n.secondWeights = add(n.secondWeights, scaledMatrix).(*mat.Dense)

	multipliedMatrix = multiply(midsError, sigInverse(activatedMids))
	dottedMatrix = dotProduct(multipliedMatrix, inputs.T())
	scaledMatrix = scale(dottedMatrix, n.speedOfNetwork)
	n.firstWeights = add(n.firstWeights, scaledMatrix).(*mat.Dense)
}



// TrainFull implements both forward and backward propagation
func (n *GoNetwork) TrainFull(data, result []float64) {
	newInputs, activatedMids, activatedOuts := n.TrainForwards(data)
	midsError, outsError := n.GetError(result, activatedOuts)
	n.TrainBackwards(newInputs, activatedMids, activatedOuts, midsError, outsError)
}

// upload saves trained hidden layer and outputs in file
func upload(n *GoNetwork) {
	mids, err := os.Create("middata.model")
	if err != nil {
		log.Fatal(err)
	} else {
		n.firstWeights.MarshalBinaryTo(mids)
	}
	defer mids.Close()

	outs, err := os.Create("outdata.model")
	if err != nil {
		log.Fatal(err)
	} else {
		n.secondWeights.MarshalBinaryTo(outs)
	}
	defer outs.Close()
}

// load sets up a neural network based on a trained data file
func load(n *GoNetwork) {
	mids, err := os.Open("middata.model")
	if err == nil {
		n.firstWeights.Reset()
		n.firstWeights.UnmarshalBinaryFrom(mids)
	}
	defer mids.Close()

	outs, err := os.Open("outdata.model")
	if err == nil {
		n.secondWeights.Reset()
		n.secondWeights.UnmarshalBinaryFrom(outs)
	}
	defer outs.Close()
}


// createRandomArray creates a new array of size n, full with random values
func createRandomArray(n int) []float64 {
	array := make([]float64, n)
	for i := range array {
		if i%2 == 0 {
			array[i] = rand.Float64()
		} else {
			array[i] = (-1) * rand.Float64()
		}
	}
	return array
}

// activateMatrix applies an activationFunction to a specified matrix
func activateMatrix(activationFunction func(x, y int, input float64) float64, matrix mat.Matrix) mat.Matrix {
	x, y := matrix.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.Apply(activationFunction, matrix)
	return resultMatrix
}

// dotProduct takes the dot product of two matrices
func dotProduct(matrixA, matrixB mat.Matrix) mat.Matrix {
	x, _ := matrixA.Dims()
	_, y := matrixB.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.Product(matrixA, matrixB)
	return resultMatrix
}

// sigmoid takes an input and returns the value post sigmoid calculation
func sigmoid(x, y int, input float64) float64 {
	return 1.0 / (1 + math.Exp(-1*input))
}

// sigInverse takes the sigmoid of (1 - sigmoid)
func sigInverse(matrix mat.Matrix) mat.Matrix {
	j, _ := matrix.Dims()
	array := make([]float64, j)
	for i := range array {
		array[i] = 1
	}
	inverseMatrix := mat.NewDense(j, 1, array)
	resultMatrix := subtract(inverseMatrix, matrix)
	return multiply(matrix, resultMatrix)
}

// subtract subtracts a matrix by another
func subtract(matrixA, matrixB mat.Matrix) mat.Matrix {
	x, y := matrixA.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.Sub(matrixA, matrixB)
	return resultMatrix
}

// add adds two matrices together
func add(matrixA, matrixB mat.Matrix) mat.Matrix {
	x, y := matrixA.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.Add(matrixA, matrixB)
	return resultMatrix
}

// scale multiplies a matrix by a scalar
func scale(matrix mat.Matrix, scalar float64) mat.Matrix {
	x, y := matrix.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.Scale(scalar, matrix)
	return resultMatrix
} 

// multiply multiplies two matrices together
func multiply(matrixA, matrixB mat.Matrix) mat.Matrix {
	x, y := matrixA.Dims()
	resultMatrix := mat.NewDense(x, y, nil)
	resultMatrix.MulElem(matrixA, matrixB)
	return resultMatrix
}


// start for numbers dataset
func numberClassification(n *GoNetwork) {
	for epochs := 0; epochs < 1; epochs++ {

		fmt.Print("Currently on epoch # ", epochs)

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

func numberPrediction(n *GoNetwork) {

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
		_, _, outputs := n.TrainForwards(inputs)
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

func predictFromImage(n *GoNetwork, path string) int {
	input := dataFromImage(path)
	_, _, output := n.TrainForwards(input)
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
	fmt.Print("Hello humans")

	// Create the neural net
	numbersNet := MakeGoNetwork(784, 200, 10, 1)
	// print(numbersNet)

	// Train the numbers
	// numberClassification(numbersNet)
	// upload(numbersNet)

	load(numbersNet)
	numberPrediction(numbersNet)

	// fmt.Println("The prediction is: ", predictFromImage(numbersNet, "numbers_png/image3.png"))

	
}
