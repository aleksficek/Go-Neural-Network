package net

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// This is the start of the neural net in golang

// Node is a data structure that represents a neural network node
type Node struct {
	value int
}

// GoNetwork is the neural network struct
type GoNetwork struct {
	speedOfNetwork float64
	inputNodes     []Node
	midNodes       []Node
	outputNodes    []Node
	firstWeights   *mat.Dense
	secondWeights  *mat.Dense
}

// MakeGoNetwork creates an instance of the neural network
func MakeGoNetwork(inputs, mids, outputs []Node, speedOfNetwork float64) {
	randomWeights1 := 1

	// neuralNet := GoNetwork{
	// 	speedOfNetwork: speedOfNetwork,
	// 	inputNodes:     inputs,
	// 	midNodes:       mids,
	// 	outputNodes:    outputs,
	// 	firstWeights: 	mat.NewDense(mids, inputs, [])
	// 	secondWeights:	mat.newDense(outputs, mids, [])
	// }
	// fmt.Print(neuralNet)
}

func main() {
	fmt.Print("Welcome to the golang neural network!")

}

func createRandomArray(n int) []int {
	array := make([]int, n)
	rand.Read(array)
	fmt.Println("Creating a random array: %v", array)
}

// sigmoid takes an input and returns the value post sigmoid calculation
func sigmoid(input float64) float64 {
	return 1.0 / (1 + math.Exp(-1*input))
}
