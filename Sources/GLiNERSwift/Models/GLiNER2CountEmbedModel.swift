import Foundation
@preconcurrency import CoreML

/// Wrapper for the GLiNER2 count embedding Core ML head.
final class GLiNER2CountEmbedModel {
    private let model: MLModel
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.countembed", qos: .userInitiated)
    private let outputFeatureName: String
    private let schemaTokenCapacity: Int
    private let maxCount: Int

    init(
        modelURL: URL,
        schemaTokenCapacity: Int,
        maxCount: Int,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) throws {
        self.schemaTokenCapacity = schemaTokenCapacity
        self.maxCount = maxCount
        let compiled = try GLiNER2CountEmbedModel.compileIfNeeded(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiled, configuration: configuration)
        self.outputFeatureName = GLiNER2CountEmbedModel.resolveOutputName(for: model)
    }

    func projectedEmbeddings(labelEmbeddings: [[Float]]) async throws -> [[[Float]]] {
        guard let hiddenSize = labelEmbeddings.first?.count, hiddenSize > 0 else {
            throw GLiNERError.invalidInput("Count embed requires label embeddings with a hidden dimension")
        }
        let labelCount = labelEmbeddings.count
        guard labelCount > 0 else {
            throw GLiNERError.invalidInput("Count embed requires at least one label embedding")
        }
        guard labelCount <= schemaTokenCapacity else {
            throw GLiNERError.invalidInput("Label embedding count (\(labelCount)) exceeds schema capacity (\(schemaTokenCapacity))")
        }

        let embeddingArray = try GLiNER2CountEmbedModel.createLabelEmbeddingArray(
            embeddings: labelEmbeddings,
            capacity: schemaTokenCapacity,
            hiddenSize: hiddenSize
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "label_embeddings": MLFeatureValue(multiArray: embeddingArray)
        ])
        let prediction = try await runPrediction(features: provider)
        guard let output = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw GLiNERError.invalidOutput("Count embed model did not produce expected output feature \(outputFeatureName)")
        }
        return try GLiNER2CountEmbedModel.decodeStructureEmbeddings(
            from: output,
            labelCount: labelCount,
            hiddenSize: hiddenSize,
            maxCount: maxCount
        )
    }
}

private extension GLiNER2CountEmbedModel {
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
        return "var_44"
    }

    func runPrediction(features: MLFeatureProvider) async throws -> MLFeatureProvider {
        let queue = predictionQueue
        let model = self.model
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let output = try model.prediction(from: features)
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    static func createLabelEmbeddingArray(
        embeddings: [[Float]],
        capacity: Int,
        hiddenSize: Int
    ) throws -> MLMultiArray {
        let labelCount = embeddings.count
        guard labelCount <= capacity else {
            throw GLiNERError.invalidInput("Label embedding count (\(labelCount)) exceeds schema capacity (\(capacity))")
        }
        let shape: [NSNumber] = [NSNumber(value: labelCount), NSNumber(value: hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        try array.withUnsafeMutableFloat32Buffer { buffer in
            guard let base = buffer.baseAddress else { return }
            var offset = 0
            for (index, row) in embeddings.enumerated() {
                guard row.count == hiddenSize else {
                    throw GLiNERError.invalidInput("Label embedding row has mismatched hidden size at index \(index)")
                }
                base.advanced(by: offset).update(from: row, count: hiddenSize)
                offset += hiddenSize
            }
        }
        return array
    }

    static func decodeStructureEmbeddings(
        from multiArray: MLMultiArray,
        labelCount: Int,
        hiddenSize: Int,
        maxCount: Int
    ) throws -> [[[Float]]] {
        let dims = multiArray.shape.map { $0.intValue }
        if dims.count == 3 {
            print("[DEBUG] CountEmbed output shape: \(dims)")
            let strideValues = multiArray.strides.map { $0.intValue }
            print("[DEBUG] CountEmbed strides: \(strideValues)")
            let zero = NSNumber(value: 0)
            let one = NSNumber(value: 1)
            let indices000 = [zero, zero, zero]
            let indices100 = [one, zero, zero]
            let indices010 = [zero, one, zero]
            let indices001 = [zero, zero, one]
            let sample00 = multiArray[indices000].floatValue
            let sample10 = multiArray[indices100].floatValue
            let sample01 = multiArray[indices010].floatValue
            let sample001 = multiArray[indices001].floatValue
            print("[DEBUG] CountEmbed samples c0l0h0=\(sample00) c1l0h0=\(sample10) c0l1h0=\(sample01) c0l0h1=\(sample001)")
        }
        guard dims.count == 3 else {
            throw GLiNERError.invalidOutput("Count embed output must be 3D, got shape \(multiArray.shape)")
        }
        let countDim = dims[0]
        let schemaDim = dims[1]
        guard countDim >= maxCount else {
            throw GLiNERError.invalidOutput("Count embed output count dimension (\(countDim)) smaller than expected maxCount \(maxCount)")
        }
        guard schemaDim >= labelCount else {
            throw GLiNERError.invalidOutput("Count embed output schema dimension (\(schemaDim)) smaller than label count \(labelCount)")
        }

        var result = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: hiddenSize),
                count: labelCount
            ),
            count: maxCount
        )
        var maxSample: (value: Float, count: Int, label: Int, hidden: Int)?
        for countIdx in 0..<maxCount {
            for schemaIdx in 0..<labelCount {
                for hiddenIdx in 0..<hiddenSize {
                    let indices: [NSNumber] = [NSNumber(value: countIdx), NSNumber(value: schemaIdx), NSNumber(value: hiddenIdx)]
                    let value = multiArray[indices].floatValue
                    result[countIdx][schemaIdx][hiddenIdx] = value
                    let absValue = abs(value)
                    if absValue > (maxSample?.value ?? 0) {
                        maxSample = (absValue, countIdx, schemaIdx, hiddenIdx)
                    }
                }
            }
        }
        if let sample = maxSample {
            print("[DEBUG] CountEmbed max value=\(sample.value) at [\(sample.count), \(sample.label), \(sample.hidden)]")
        }
        return result
    }
}
