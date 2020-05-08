package main // This will need to change and become abstracted

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

func parseData(path string) {
	// open file
	
}

func main() {
	fmt.Print("Hello humans")

	// Create the neural net
	numbersNet := MakeGoNetwork(784, 200, 10, 1)
	// print(numbersNet)
}

