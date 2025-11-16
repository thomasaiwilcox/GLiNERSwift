import Accelerate
import Foundation

/// Simple fully-connected layer with bias, backed by Accelerate for dot products.
struct LinearLayer {
    let weights: [Float]    // Stored row-major [outFeatures, inFeatures]
    let bias: [Float]
    let inFeatures: Int
    let outFeatures: Int

    init(weights: [Float], bias: [Float], inFeatures: Int, outFeatures: Int) {
        precondition(weights.count == inFeatures * outFeatures,
                     "Weight matrix shape mismatch: expected \(outFeatures)x\(inFeatures) entries")
        precondition(bias.count == outFeatures, "Bias vector size mismatch")
        self.weights = weights
        self.bias = bias
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
    }

    /// Apply the linear layer to a single vector.
    func apply(to input: [Float]) -> [Float] {
        precondition(input.count == inFeatures, "Input width mismatch: expected \(inFeatures), got \(input.count)")

        var output = bias
        output.withUnsafeMutableBufferPointer { outPtr in
            weights.withUnsafeBufferPointer { weightPtr in
                input.withUnsafeBufferPointer { inputPtr in
                    for outIdx in 0..<outFeatures {
                        let weightOffset = outIdx * inFeatures
                        var dot: Float = 0
                        vDSP_dotpr(
                            inputPtr.baseAddress!, 1,
                            weightPtr.baseAddress!.advanced(by: weightOffset), 1,
                            &dot,
                            vDSP_Length(inFeatures)
                        )
                        outPtr[outIdx] += dot
                    }
                }
            }
        }
        return output
    }

    /// Apply the linear layer to a batch of vectors.
    func apply(to inputs: [[Float]]) -> [[Float]] {
        inputs.map { apply(to: $0) }
    }
}

/// Two-layer feed-forward block matching GLiNER's projection helper (Linear → ReLU → Dropout → Linear).
struct FeedForwardLayer {
    private let first: LinearLayer
    private let second: LinearLayer

    init(first: LinearLayer, second: LinearLayer) {
        precondition(first.outFeatures == second.inFeatures,
                     "FeedForward dimension mismatch: \(first.outFeatures) != \(second.inFeatures)")
        self.first = first
        self.second = second
    }

    func apply(to inputs: [[Float]]) -> [[Float]] {
        let hidden = first.apply(to: inputs).map { relu($0) }
        return second.apply(to: hidden)
    }

    func apply(to input: [Float]) -> [Float] {
        let hidden = relu(first.apply(to: input))
        return second.apply(to: hidden)
    }
}

/// Element-wise ReLU
private func relu(_ values: [Float]) -> [Float] {
    var result = values
    var threshold: Float = 0
    vDSP_vthres(result, 1, &threshold, &result, 1, vDSP_Length(result.count))
    return result
}
