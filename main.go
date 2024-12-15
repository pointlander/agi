// Copyright 2024 The AGI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

const (
	// Size is the number of histograms
	Size = 11
	// Rows is the number of rows the matrix has
	Rows = 4 * Size
	// EndLine is the end of the line
	EndLine = 10
	// EndBlock is the endof a block of lines
	EndBlock = 11
)

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	return sets
}

// Markov2 is a markov model
type Markov2 [2]byte

// Markov is a 3rd order markov model
type Markov3 [3]byte

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]uint16
	Buffer [1024]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

// HistogramSet is a histogram set
type HistogramSet struct {
	Histograms [Size]Histogram
}

// NewHistogramSet makes a new histogram set
func NewHistogramSet() HistogramSet {
	h := HistogramSet{}
	h.Histograms[0] = NewHistogram(1)
	h.Histograms[1] = NewHistogram(2)
	h.Histograms[2] = NewHistogram(4)
	h.Histograms[3] = NewHistogram(8)
	h.Histograms[4] = NewHistogram(16)
	h.Histograms[5] = NewHistogram(32)
	h.Histograms[6] = NewHistogram(64)
	h.Histograms[7] = NewHistogram(128)
	h.Histograms[8] = NewHistogram(256)
	h.Histograms[9] = NewHistogram(512)
	h.Histograms[10] = NewHistogram(1024)
	return h
}

// Mixer mixes several histograms together
type Mixer struct {
	Markov2 Markov2
	Markov3 Markov3
	Set     HistogramSet
	Set1    [256]HistogramSet
	Set2    map[Markov2]*HistogramSet
	Set3    map[Markov3]*HistogramSet
}

// NewMixer makes a new mixer
func NewMixer() Mixer {
	m := Mixer{
		Set: NewHistogramSet(),
	}
	for i := range m.Set1 {
		m.Set1[i] = NewHistogramSet()
	}
	m.Set2 = make(map[Markov2]*HistogramSet)
	m.Set3 = make(map[Markov3]*HistogramSet)
	return m
}

// MixFloat64 mixes the histograms outputting float64
func (m *Mixer) Mix() [256]float64 {
	mix := [256]float64{}
	x := NewMatrix(256, Rows)
	for i := range m.Set.Histograms {
		sum := 0.0
		for _, v := range m.Set.Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Set.Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	for i := range m.Set1[m.Markov2[0]].Histograms {
		sum := 0.0
		for _, v := range m.Set1[m.Markov2[0]].Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Set1[m.Markov2[0]].Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	for i := range m.Set2[m.Markov2].Histograms {
		sum := 0.0
		for _, v := range m.Set2[m.Markov2].Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Set2[m.Markov2].Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	for i := range m.Set3[m.Markov3].Histograms {
		sum := 0.0
		for _, v := range m.Set3[m.Markov3].Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Set3[m.Markov3].Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}

	y := SelfAttention(x, x, x).Sum()
	sum := 0.0
	for _, v := range y.Data {
		sum += v
	}
	for i := range mix {
		mix[i] = y.Data[i] / sum
	}
	return mix
}

// Add adds a symbol to a mixer
func (m *Mixer) Add(s byte) {
	for i := range m.Set.Histograms {
		m.Set.Histograms[i].Add(s)
	}
	m.Markov2[1] = m.Markov2[0]
	m.Markov2[0] = s
	m.Markov3[2] = m.Markov3[1]
	m.Markov3[1] = m.Markov3[0]
	m.Markov3[0] = s
	for i := range m.Set1[m.Markov2[0]].Histograms {
		m.Set1[m.Markov2[0]].Histograms[i].Add(s)
	}
	if m.Set2[m.Markov2] == nil {
		set := NewHistogramSet()
		m.Set2[m.Markov2] = &set
	}
	for i := range m.Set2[m.Markov2].Histograms {
		m.Set2[m.Markov2].Histograms[i].Add(s)
	}
	if m.Set3[m.Markov3] == nil {
		set := NewHistogramSet()
		m.Set3[m.Markov3] = &set
	}
	for i := range m.Set3[m.Markov3].Histograms {
		m.Set3[m.Markov3].Histograms[i].Add(s)
	}
}

// TXT is a context
type TXT struct {
	Vector [256]float64
	Symbol byte
}

// CS is the cosine similarity
func (t *TXT) CS(vector *[256]float64) float64 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := vector[i], float64(t.Vector[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

func main() {
	s, m := Load(), NewMixer()
	set := s[0]
	encoding := make([]byte, 0, 8)
	for i := range set.Train {
		for j := range set.Train[i].Input {
			encoding = append(encoding, set.Train[i].Input[j]...)
			encoding = append(encoding, EndLine)
		}
		encoding = append(encoding, EndBlock)
		for j := range set.Train[i].Output {
			encoding = append(encoding, set.Train[i].Output[j]...)
			encoding = append(encoding, EndLine)
		}
		encoding = append(encoding, EndBlock)
	}
	for j := range set.Test[0].Input {
		encoding = append(encoding, set.Test[0].Input[j]...)
		encoding = append(encoding, EndLine)
	}
	encoding = append(encoding, EndBlock)
	txts := make([]TXT, 0, 8)
	for i := range encoding[:len(encoding)-1] {
		if i > 0 {
			txts = append(txts, TXT{
				Vector: m.Mix(),
				Symbol: encoding[i+1],
			})
		}
		m.Add(encoding[i])
	}

	rng := rand.New(rand.NewSource(1))
	{
		set := tf64.NewSet()
		set.Add("w1", 256, 256)
		set.Add("b1", 256)
		set.Add("w2", 256, 256)
		set.Add("b2", 256)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
		}

		others := tf64.NewSet()
		others.Add("input", 256)
		others.Add("output", 256)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
		l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
		loss := tf64.Quadratic(l2, others.Get("output"))

		points := make(plotter.XYs, 0, 8)
		for i := 0; i < 33*1024; i++ {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(i+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			others.Zero()
			index := rng.Intn(len(txts))
			input := others.ByName["input"].X
			for j := range input {
				input[j] = txts[index].Vector[j]
			}
			output := others.ByName["output"].X
			for j := range output {
				output[j] = 0
			}
			output[txts[index].Symbol] = 1

			set.Zero()
			cost := tf64.Gradient(loss).X[0]

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
			fmt.Println(cost)
		}

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
		if err != nil {
			panic(err)
		}
	}

	m.Add(encoding[len(encoding)-1])
	solution := make([]byte, 0, 8)
	for {
		vector, max, symbol := m.Mix(), -1.0, byte(0)
		for i := range txts {
			s := txts[i].CS(&vector)
			if s > max {
				max, symbol = s, txts[i].Symbol
			}
		}
		solution = append(solution, symbol)
		m.Add(symbol)
		if symbol == EndBlock {
			break
		}
	}
	for j := range set.Test[0].Output {
		fmt.Println(set.Test[0].Output[j])
	}
	fmt.Println()
	for i := range solution {
		if solution[i] == EndLine {
			fmt.Println()
			continue
		} else if solution[i] == EndBlock {
			fmt.Println()
			break
		}
		fmt.Printf("%d ", solution[i])
	}
}
