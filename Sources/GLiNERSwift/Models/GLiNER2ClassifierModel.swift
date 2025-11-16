import Foundation
@preconcurrency import CoreML

/// Wrapper for the GLiNER2 classifier Core ML head.
final class GLiNER2ClassifierModel {
    private let model: MLModel
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.classifier", qos: .userInitiated)
    private let outputFeatureName: String

    init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let compiled = try GLiNER2ClassifierModel.compileIfNeeded(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiled, configuration: configuration)
        self.outputFeatureName = GLiNER2ClassifierModel.resolveOutputName(for: model)
    }

    func logits(schemaEmbeddings: [[Float]]) async throws -> MLMultiArray {
        guard let hiddenSize = schemaEmbeddings.first?.count, hiddenSize > 0 else {
            throw GLiNERError.invalidInput("Classifier embeddings must include a hidden dimension")
        }
        let promptCount = schemaEmbeddings.count
        guard promptCount > 0 else {
            throw GLiNERError.invalidInput("Classifier requires at least one prompt embedding")
        }
        let embeddings = try GLiNER2ClassifierModel.createSchemaEmbeddingArray(
            embeddings: schemaEmbeddings,
            promptCount: promptCount,
            hiddenSize: hiddenSize
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "schema_embeddings": MLFeatureValue(multiArray: embeddings)
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
        guard let logits = prediction.featureValue(for: featureName)?.multiArrayValue else {
            throw GLiNERError.invalidOutput("Classifier did not produce expected logits output \(featureName)")
        }
        return logits
    }
}

private extension GLiNER2ClassifierModel {
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

    static func createSchemaEmbeddingArray(
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
                    throw GLiNERError.invalidInput("Schema embedding row has mismatched hidden size")
                }
                base.advanced(by: offset).update(from: row, count: hiddenSize)
                offset += hiddenSize
            }
        }
        return array
    }
}
