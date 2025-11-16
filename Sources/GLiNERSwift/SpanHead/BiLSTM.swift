import Accelerate
import Foundation

struct LSTMWeightsSet {
    let weightIH: [Float]
    let weightHH: [Float]
    let bias: [Float]
}

/// Minimal single-layer bidirectional LSTM used to mirror GLiNER's contextual word encoder.
final class BiLSTM {
    private let inputSize: Int
    private let hiddenSize: Int
    private let forwardWeights: LSTMWeightsSet
    private let backwardWeights: LSTMWeightsSet?

    init(inputSize: Int, hiddenSize: Int, forward: LSTMWeightsSet, backward: LSTMWeightsSet?) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.forwardWeights = forward
        self.backwardWeights = backward
    }

    func apply(to sequence: [[Float]]) -> [[Float]] {
        guard !sequence.isEmpty else { return [] }
        let forwardOutputs = run(direction: forwardWeights, inputs: sequence, reverse: false)
        let backwardOutputs = backwardWeights.flatMap { run(direction: $0, inputs: sequence, reverse: true) }
        var combined: [[Float]] = []
        combined.reserveCapacity(sequence.count)
        for idx in 0..<sequence.count {
            let forwardHidden = forwardOutputs[idx]
            if let backwardHidden = backwardOutputs?[idx] {
                combined.append(forwardHidden + backwardHidden)
            } else {
                combined.append(forwardHidden)
            }
        }
        return combined
    }

    private func run(direction: LSTMWeightsSet, inputs: [[Float]], reverse: Bool) -> [[Float]] {
        let timeSteps = inputs.count
        var outputs = Array(repeating: [Float](repeating: 0, count: hiddenSize), count: timeSteps)
        var h = [Float](repeating: 0, count: hiddenSize)
        var c = [Float](repeating: 0, count: hiddenSize)
        let indices: [Int] = reverse
            ? Array(stride(from: timeSteps - 1, through: 0, by: -1))
            : Array(0..<timeSteps)

        for (step, inputIndex) in indices.enumerated() {
            let inputVector = inputs[inputIndex]
            precondition(inputVector.count == inputSize, "LSTM input width mismatch: expected \(inputSize), got \(inputVector.count)")

            var gates = direction.bias
            accumulate(weights: direction.weightIH, rows: hiddenSize * 4, cols: inputSize, vector: inputVector, into: &gates)
            accumulate(weights: direction.weightHH, rows: hiddenSize * 4, cols: hiddenSize, vector: h, into: &gates)

            var inputGate = [Float](repeating: 0, count: hiddenSize)
            var forgetGate = [Float](repeating: 0, count: hiddenSize)
            var cellGate = [Float](repeating: 0, count: hiddenSize)
            var outputGate = [Float](repeating: 0, count: hiddenSize)

            applySigmoid(source: gates, offset: 0, result: &inputGate)
            applySigmoid(source: gates, offset: hiddenSize, result: &forgetGate)
            applyTanh(source: gates, offset: hiddenSize * 2, result: &cellGate)
            applySigmoid(source: gates, offset: hiddenSize * 3, result: &outputGate)

            for i in 0..<hiddenSize {
                let candidate = cellGate[i]
                let forget = forgetGate[i]
                let inputValue = inputGate[i]
                c[i] = forget * c[i] + inputValue * candidate
                let activatedCell = tanhf(c[i])
                h[i] = outputGate[i] * activatedCell
            }

            let targetIndex = reverse ? (timeSteps - 1 - step) : step
            outputs[targetIndex] = h
        }
        return outputs
    }

    private func accumulate(weights: [Float], rows: Int, cols: Int, vector: [Float], into output: inout [Float]) {
        precondition(weights.count == rows * cols, "Matrix shape mismatch")
        vector.withUnsafeBufferPointer { vectorPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                weights.withUnsafeBufferPointer { weightPtr in
                    for row in 0..<rows {
                        var dot: Float = 0
                        let weightOffset = weightPtr.baseAddress! + row * cols
                        vDSP_dotpr(vectorPtr.baseAddress!, 1, weightOffset, 1, &dot, vDSP_Length(cols))
                        outPtr[row] += dot
                    }
                }
            }
        }
    }

    private func applySigmoid(source: [Float], offset: Int, result: inout [Float]) {
        for idx in 0..<hiddenSize {
            result[idx] = 1.0 / (1.0 + exp(-source[offset + idx]))
        }
    }

    private func applyTanh(source: [Float], offset: Int, result: inout [Float]) {
        for idx in 0..<hiddenSize {
            result[idx] = tanhf(source[offset + idx])
        }
    }
}

private extension Array where Element == Float {
    static func +(lhs: [Float], rhs: [Float]) -> [Float] {
        var combined = lhs
        combined.append(contentsOf: rhs)
        return combined
    }
}
