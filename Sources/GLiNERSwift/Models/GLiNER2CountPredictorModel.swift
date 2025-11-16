import Foundation
@preconcurrency import CoreML

/// Wrapper for the GLiNER2 count predictor Core ML head.
final class GLiNER2CountPredictorModel {
    private let model: MLModel
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.countpredictor", qos: .userInitiated)
    private let outputFeatureName: String

    init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let compiled = try GLiNER2CountPredictorModel.compileIfNeeded(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiled, configuration: configuration)
        self.outputFeatureName = GLiNER2CountPredictorModel.resolveOutputName(for: model)
    }

    func predict(promptEmbeddings: [[Float]]) async throws -> MLMultiArray {
        guard let hiddenSize = promptEmbeddings.first?.count, hiddenSize > 0 else {
            throw GLiNERError.invalidInput("Count predictor embeddings must include a hidden dimension")
        }
        let promptCount = promptEmbeddings.count
        guard promptCount > 0 else {
            throw GLiNERError.invalidInput("Count predictor requires at least one prompt embedding")
        }
        let embeddings = try GLiNER2CountPredictorModel.createPromptEmbeddingArray(
            embeddings: promptEmbeddings,
            promptCount: promptCount,
            hiddenSize: hiddenSize
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "prompt_embeddings": MLFeatureValue(multiArray: embeddings)
        ])
        let queue = predictionQueue
        let model = self.model
        let featureName = outputFeatureName
        let prediction = try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try model.prediction(from: provider)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        guard let output = prediction.featureValue(for: featureName)?.multiArrayValue else {
            throw GLiNERError.invalidOutput("Count predictor did not produce expected output \(featureName)")
        }
        return output
    }
}

private extension GLiNER2CountPredictorModel {
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
        return "var_11"
    }

    static func createPromptEmbeddingArray(
        embeddings: [[Float]],
        promptCount: Int,
        hiddenSize: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [NSNumber(value: promptCount), NSNumber(value: hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        try array.withUnsafeMutableFloat32Buffer { buffer in
            guard let base = buffer.baseAddress else { return }
            var offset = 0
            for row in embeddings {
                guard row.count == hiddenSize else {
                    throw GLiNERError.invalidInput("Prompt embedding row has mismatched hidden size")
                }
                base.advanced(by: offset).update(from: row, count: hiddenSize)
                offset += hiddenSize
            }
        }
        return array
    }
}
