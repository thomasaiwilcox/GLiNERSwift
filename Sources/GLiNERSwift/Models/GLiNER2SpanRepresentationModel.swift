import Foundation
@preconcurrency import CoreML

/// Runs the GLiNER2 span representation Core ML model.
final class GLiNER2SpanRepresentationModel {
    private let model: MLModel
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.spanrep", qos: .userInitiated)
    private let outputFeatureName: String

    init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let compiledURL = try GLiNER2SpanRepresentationModel.compileIfNeeded(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.outputFeatureName = GLiNER2SpanRepresentationModel.resolveOutputName(for: model)
    }

    func spanEmbeddings(
        tokenEmbeddings: [[Float]],
        spanIndices: [[Int32]]
    ) async throws -> [[[Float]]] {
        guard let hiddenSize = tokenEmbeddings.first?.count, hiddenSize > 0 else {
            throw GLiNERError.invalidInput("Token embeddings must include at least one hidden dimension")
        }
        let sequenceLength = tokenEmbeddings.count
        guard sequenceLength > 0 else {
            throw GLiNERError.invalidInput("Token embeddings must include at least one token")
        }
        guard spanIndices.count > 0, spanIndices.first?.count == 2 else {
            throw GLiNERError.invalidInput("Span indices must include [start, end] pairs")
        }
        guard spanIndices.count % sequenceLength == 0 else {
            throw GLiNERError.invalidInput("Span indices count must be divisible by sequence length")
        }
        let maxWidth = spanIndices.count / sequenceLength

        let tokenArray = try GLiNER2SpanRepresentationModel.createTokenEmbeddingArray(
            embeddings: tokenEmbeddings,
            sequenceLength: sequenceLength,
            hiddenSize: hiddenSize
        )
        let spanArray = try GLiNER2SpanRepresentationModel.createSpanIndexArray(
            spanIndices: spanIndices
        )

        let prediction = try await runPrediction(
            tokenEmbeddings: tokenArray,
            spanIndices: spanArray
        )
        guard let spanValues = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw GLiNERError.invalidOutput("SpanRep model did not produce expected output feature \(outputFeatureName)")
        }
        return try GLiNER2SpanRepresentationModel.decodeSpanEmbeddings(
            from: spanValues,
            sequenceLength: sequenceLength,
            maxWidth: maxWidth,
            hiddenSize: hiddenSize
        )
    }

    private func runPrediction(
        tokenEmbeddings: MLMultiArray,
        spanIndices: MLMultiArray
    ) async throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "token_embeddings": MLFeatureValue(multiArray: tokenEmbeddings),
            "span_indices": MLFeatureValue(multiArray: spanIndices)
        ])
        let queue = predictionQueue
        let model = self.model
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let output = try model.prediction(from: provider)
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

private extension GLiNER2SpanRepresentationModel {
    static func compileIfNeeded(at url: URL) throws -> URL {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "mlmodelc":
            return url
        case "mlmodel", "mlpackage":
            return try MLModel.compileModel(at: url)
        default:
            return url
        }
    }

    static func resolveOutputName(for model: MLModel) -> String {
        if let first = model.modelDescription.outputDescriptionsByName.keys.first {
            return first
        }
        return "var_74"
    }

    static func createTokenEmbeddingArray(
        embeddings: [[Float]],
        sequenceLength: Int,
        hiddenSize: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sequenceLength), NSNumber(value: hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        try array.withUnsafeMutableFloat32Buffer { buffer in
            guard let base = buffer.baseAddress else { return }
            var offset = 0
            for row in embeddings {
                guard row.count == hiddenSize else {
                    throw GLiNERError.invalidInput("Token embedding row has mismatched hidden size")
                }
                base.advanced(by: offset).update(from: row, count: hiddenSize)
                offset += hiddenSize
            }
        }
        return array
    }

    static func createSpanIndexArray(spanIndices: [[Int32]]) throws -> MLMultiArray {
        let spanCount = spanIndices.count
        let shape: [NSNumber] = [1, NSNumber(value: spanCount), 2]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        var offset = 0
        for pair in spanIndices {
            guard pair.count == 2 else {
                throw GLiNERError.invalidInput("Each span index entry must contain exactly two values")
            }
            pointer[offset] = pair[0]
            pointer[offset + 1] = pair[1]
            offset += 2
        }
        return array
    }

    static func decodeSpanEmbeddings(
        from multiArray: MLMultiArray,
        sequenceLength: Int,
        maxWidth: Int,
        hiddenSize: Int
    ) throws -> [[[Float]]] {
        guard multiArray.shape.count == 4 else {
            throw GLiNERError.invalidOutput("Expected 4D span embeddings tensor, got \(multiArray.shape.count)D")
        }
        guard multiArray.shape[0].intValue == 1 else {
            throw GLiNERError.invalidOutput("Span embeddings batch dimension must be 1")
        }
        var result = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: hiddenSize),
                count: maxWidth
            ),
            count: sequenceLength
        )
        for seqIdx in 0..<sequenceLength {
            for widthIdx in 0..<maxWidth {
                for hiddenIdx in 0..<hiddenSize {
                    let indices: [NSNumber] = [0, NSNumber(value: seqIdx), NSNumber(value: widthIdx), NSNumber(value: hiddenIdx)]
                    result[seqIdx][widthIdx][hiddenIdx] = multiArray[indices].floatValue
                }
            }
        }
        return result
    }
}
