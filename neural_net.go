package net

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// This is the start of the neural net in golang!

// Node is a data structure that represents a neural network node
type Node struct {
	value int
}

// GoNetwork is the neural network struct
type GoNetwork struct {
	speedOfNetwork float64
	inputs         int
	mids           int
	outputs        int
	firstWeights   *mat.Dense
	secondWeights  *mat.Dense
}

// MakeGoNetwork creates an instance of the neural network
func MakeGoNetwork(inputs, mids, outputs int, speedOfNetwork float64) {

	neuralNet := GoNetwork{
		speedOfNetwork: speedOfNetwork,
		inputs:         inputs,
		mids:           mids,
		outputs:        outputs,
		firstWeights:   mat.NewDense(mids, inputs, createRandomArray(mids)),
		secondWeights:  mat.NewDense(outputs, mids, createRandomArray(outputs)),
	}
	fmt.Print(neuralNet)
}

// TrainOnce performs a single iteration of applying weights and activating function to network
func (n *GoNetwork) TrainOnce(data []float64) mat.Matrix {
	newInputs := mat.NewDense(n.inputs, 1, data)
	newMids := dotProduct(n.firstWeights, newInputs)
	newMidsActivated := activateMatrix(sigmoid, newMids)
	newOuts := dotProduct(n.secondWeights, newMidsActivated)
	return activateMatrix(sigmoid, newOuts)
}

func main() {
	fmt.Print("Welcome to the golang neural network!")
	myNet := &GoNetwork{
		speedOfNetwork: 13,
		inputs:         17,
		mids:           15,
		outputs:        3,
	}

	data := [...]float64{13, 14, 15, 13, 17, 2, 4, 5, 7}
	firstPrediction := myNet.TrainOnce(data)
}

// createRandomArray creates a new array of size n, full with random values
func createRandomArray(n int) []float64 {
	fmt.Println("Creating a random array!")
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
