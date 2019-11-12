package net

import "math"

// This is the start of the neural net in golang

type GoNetwork struct {
	input int
}

func sigmoid(input float64) float64 {
	return 1.0 / (1 + math.Exp(-1*input))
}
