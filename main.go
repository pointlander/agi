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
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"github.com/fxsjy/RF.go/RF"
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
	// Buffer is the buffer size
	Buffer = 1 << Size
	// T is the temperature factor
	T = 1
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
	Buffer [Buffer]byte
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
	for i := range h.Histograms {
		h.Histograms[i] = NewHistogram(1 << i)
	}
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

// Copy copies a mixer
func (m *Mixer) Copy() Mixer {
	n := Mixer{
		Markov2: m.Markov2,
		Markov3: m.Markov3,
		Set:     m.Set,
		Set1:    m.Set1,
	}
	n.Set2 = make(map[Markov2]*HistogramSet)
	for k, v := range m.Set2 {
		h := NewHistogramSet()
		h = *v
		n.Set2[k] = &h
	}
	n.Set3 = make(map[Markov3]*HistogramSet)
	for k, v := range m.Set3 {
		h := NewHistogramSet()
		h = *v
		n.Set3[k] = &h
	}
	return n
}

// Mix mixes the histograms outputting float64
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

// Computes the softmax of a vector
func Softmax(vector *[256]float64, T float64) {
	max := 0.0
	for _, v := range vector {
		v /= T
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	values := [256]float64{}
	for j, value := range vector {
		values[j] = math.Exp(value/T - s)
		sum += values[j]
	}
	for j, value := range values {
		vector[j] = value / sum
	}
}

// RandVector returns a random vector
func RandVector(seed int64) (vector [256]float64) {
	rng := rand.New(rand.NewSource(seed))
	for i := range vector {
		vector[i] = math.Abs(rng.NormFloat64())
	}
	Softmax(&vector, T)
	return vector
}

// MixRand mixes the histograms and a random vector outputting float64
func (m *Mixer) MixRand(seed int64) [256]float64 {
	mix := [256]float64{}
	x := NewMatrix(256, Rows+1)
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
	vec := RandVector(seed)
	for _, v := range vec {
		x.Data = append(x.Data, v)
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

// Neural is a neural network
type Neural struct {
	Set    tf64.Set
	Others tf64.Set
	L1     tf64.Meta
	L2     tf64.Meta
	Loss   tf64.Meta
}

// Learn learn a neural network
func Learn(txts []TXT) Neural {
	rng := rand.New(rand.NewSource(1))
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
		if i%1024 == 0 {
			fmt.Println(cost)
		}
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
	return Neural{
		Set:    set,
		Others: others,
		L1:     l1,
		L2:     l2,
		Loss:   loss,
	}
}

// Inference performs inference of the neural network
func (n *Neural) Inference(input [256]float64) int {
	symbol, max := 0, 0.0
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L2(func(a *tf64.V) bool {
		for i, v := range a.X {
			if v > max {
				max, symbol = v, i
			}
		}
		return true
	})

	return symbol
}

// NewForest builds a random forest
func NewForest(txts []TXT) *RF.Forest {
	inputs := make([][]interface{}, 0)
	targets := make([]string, 0)
	for i := range txts {
		row := make([]interface{}, 0)
		for j := range txts[i].Vector {
			row = append(row, txts[i].Vector[j])
		}
		inputs = append(inputs, row)
		targets = append(targets, fmt.Sprintf("%d", txts[i].Symbol))
	}
	return RF.BuildForest(inputs, targets, 1024, len(inputs), 256)
}

// ForestInference run inference on random forest
func ForestInference(rf *RF.Forest, vector [256]float64) int {
	input := make([]interface{}, 0, 8)
	for _, v := range vector {
		input = append(input, v)
	}
	y := rf.Predicate(input)
	i, err := strconv.Atoi(y)
	if err != nil {
		panic(err)
	}
	return i
}

// Top is a top
type Top struct {
	TXT  *TXT
	Rank float64
}

// Top8 is the top 8
type Top8 [8]Top

// Top8Sort sorts the top 8 results
func (t *Top8) Top8Sort(txts []TXT, vector *[256]float64) {
	for i := range txts {
		s := txts[i].CS(vector)
		for j := range t {
			if s > t[j].Rank {
				next := t[j]
				for k := j + 1; k < len(t); k++ {
					t[k], next = next, t[k]
				}
				t[j].Rank, t[j].TXT = s, &txts[i]
				break
			}
		}
	}
}

// Depth is the minimax search depth
const Depth = 5

// Max is the minimax max function
func Max(txts []TXT, depth int, action byte, m *Mixer) float64 {
	if depth >= Depth {
		cp := m.Copy()
		cp.Add(action)
		histogram, vector, top := [256]int{}, cp.Mix(), Top8{}

		top.Top8Sort(txts, &vector)
		for i := range top {
			histogram[top[i].TXT.Symbol]++
		}
		sum := 0.0
		for i := range histogram {
			sum += float64(histogram[i])
		}
		e := 0.0
		for _, v := range histogram {
			if v == 0 {
				continue
			}
			e += float64(v) * math.Log(float64(v)/sum) / sum
		}
		return -e
		/*avg, count := 0.0, 0.0
		for i := range histogram {
			avg += float64(histogram[i]) / sum
			count++
		}
		avg /= count
		stddev := 0.0
		for i := range histogram {
			diff := (float64(histogram[i]) / sum) - avg
			stddev += diff * diff
		}
		stddev = math.Sqrt(stddev / count)
		return stddev*/
	}
	cp := m.Copy()
	cp.Add(action)
	histogram, vector, top := [256]int{}, cp.Mix(), Top8{}
	top.Top8Sort(txts, &vector)
	for i := range top {
		histogram[top[i].TXT.Symbol]++
	}
	max := 0.0
	for i, v := range histogram {
		if v > 0 {
			x := Min(txts, depth+1, byte(i), &cp)
			if x > max {
				max = x
			}
		}
	}
	return max
}

// Min is the minimax min function
func Min(txts []TXT, depth int, action byte, m *Mixer) float64 {
	if depth >= Depth {
		cp := m.Copy()
		cp.Add(action)
		histogram, vector, top := [256]int{}, cp.Mix(), Top8{}
		top.Top8Sort(txts, &vector)
		for i := range top {
			histogram[top[i].TXT.Symbol]++
		}
		sum := 0.0
		for i := range histogram {
			sum += float64(histogram[i])
		}
		e := 0.0
		for _, v := range histogram {
			if v == 0 {
				continue
			}
			e += float64(v) * math.Log(float64(v)/sum) / sum
		}
		return -e
		/*avg, count := 0.0, 0.0
		for i := range histogram {
			avg += float64(histogram[i]) / sum
			count++
		}
		avg /= count
		stddev := 0.0
		for i := range histogram {
			diff := (float64(histogram[i]) / sum) - avg
			stddev += diff * diff
		}
		stddev = math.Sqrt(stddev / count)
		return stddev*/
	}
	cp := m.Copy()
	cp.Add(action)
	histogram, vector, top := [256]int{}, cp.Mix(), Top8{}
	top.Top8Sort(txts, &vector)
	for i := range top {
		histogram[top[i].TXT.Symbol]++
	}
	min := math.MaxFloat64
	for i, v := range histogram {
		if v > 0 {
			x := Max(txts, depth+1, byte(i), &cp)
			if x < min {
				min = x
			}
		}
	}
	return min
}

func main() {
	s, m := Load(), NewMixer()
	set := s[0]
	encoding := make([]byte, 0, 8)
	for i := range set.Train {
		for j := range set.Train[i].Input {
			encoding = append(encoding, set.Train[i].Input[j]...)
			if j < len(set.Train[i].Input)-1 {
				encoding = append(encoding, EndLine)
			}
		}
		encoding = append(encoding, EndBlock)
		for j := range set.Train[i].Output {
			encoding = append(encoding, set.Train[i].Output[j]...)
			if j < len(set.Train[i].Output)-1 {
				encoding = append(encoding, EndLine)
			}
		}
		encoding = append(encoding, EndBlock)
	}
	for j := range set.Test[0].Input {
		encoding = append(encoding, set.Test[0].Input[j]...)
		if j < len(set.Test[0].Input)-1 {
			encoding = append(encoding, EndLine)
		}
	}
	encoding = append(encoding, EndBlock)
	txts := make([]TXT, 0, 8)
	for i := range encoding[:len(encoding)-1] {
		m.Add(encoding[i])
		if i > 0 {
			txts = append(txts, TXT{
				Vector: m.Mix(),
				Symbol: encoding[i+1],
			})
		}
	}
	//neural := Learn(txts)
	m.Add(encoding[len(encoding)-1])
	solution := make([]byte, 0, 8)
	for {
		histogram, vector, top := [256]int{}, m.Mix(), Top8{}
		top.Top8Sort(txts, &vector)
		for i := range top {
			histogram[top[i].TXT.Symbol]++
		}
		action, max := byte(0), 0.0
		for i, v := range histogram {
			if v > 0 {
				x := Max(txts, 0, byte(i), &m)
				if x > max {
					action, max = byte(i), x
				}
			}
		}
		solution = append(solution, action)
		//sym := neural.Inference(vector)
		//fmt.Println(sym)
		fmt.Println(action)
		m.Add(action)
		if action == EndBlock {
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
			fmt.Println()
			break
		}
		fmt.Printf("%d ", solution[i])
	}
}
