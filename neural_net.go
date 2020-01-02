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

// TrainForwards performs a single iteration of forward propagation 
func (n *GoNetwork) TrainForwards(data []float64) (mat.Matrix, mat.Matrix, mat.Matrix) {
	newInputs := mat.NewDense(n.inputs, 1, data)
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
	dottedMatrix := dot(multipliedMatrix, activatedMids.T())
	scaledMatrix := scale(n.speedOfNetwork, dottedMatrix)
	n.secondWeights = add(n.secondWeights, scaledMatrix).(*mat.Dense)

	multipliedMatrix = multiply(midsError, sigInverse(activatedMids))
	dottedMatrix = dot(multipliedMatrix, inputs.T())
	scaledMatrix = scale(n.speedOfNetwork, dottedMatrix)
	n.firstWeights = add(n.firstWeights, scaledMatrix).(*mat.Dense)
}



// TrainFull implements both forward and backward propagation
func (n *GoNetwork) TrainFull(data, result []float64) {
	newInputs, activatedMids, activatedOuts := n.TrainForwards(data)
	midsError, outsError := n.GetError(result, activatedOuts)

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

// sigInverse takes the sigmoid of (1 - sigmoid)
func sigInverse(matrix mat.Matrix) mat.Matrix {
	j, _ := matrix.Dims()
	array := make([]float64, j)
	for i := range array {
		array[i] = 1
	}
	inverseMatrix := mat.NewDense(j, 1, array)
	inverseMatrix = subtract(inverseMatrix, matrix)
	return multiply(matrix, inverseMatrix)
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