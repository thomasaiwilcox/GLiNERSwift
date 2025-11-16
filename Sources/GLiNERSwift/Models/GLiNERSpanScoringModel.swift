import Foundation
@preconcurrency import CoreML

struct SpanScoringLimits {
    let maxWordCount: Int
    let maxPromptCount: Int
}

/// Wrapper around the Core ML span scoring head.
final class GLiNERSpanScoringModel {
    private let model: MLModel
    private let outputFeatureName: String
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.scoring.prediction", qos: .userInitiated)
    let limits: SpanScoringLimits
    
    init(bundle: Bundle = .module, limits: SpanScoringLimits = .init(maxWordCount: 256, maxPromptCount: 64)) throws {
        self.limits = limits
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        guard let modelURL = bundle.url(
            forResource: "GLiNERSpanScorer",
            withExtension: "mlpackage",
            subdirectory: "Resources"
        ) else {
            throw GLiNERError.modelNotFound("GLiNERSpanScorer.mlpackage not found in bundle. Re-run Scripts/convert_to_coreml.py to generate it.")
        }
        let compiledURL = try GLiNERSpanScoringModel.compileIfNeeded(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.outputFeatureName = GLiNERSpanScoringModel.resolveOutputFeatureName(for: self.model)
    }
    
    init(modelURL: URL, computeUnits: MLComputeUnits = .cpuOnly, limits: SpanScoringLimits) throws {
        self.limits = limits
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let compiledURL = try GLiNERSpanScoringModel.compileIfNeeded(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.outputFeatureName = GLiNERSpanScoringModel.resolveOutputFeatureName(for: self.model)
    }
    
    func score(
        wordEmbeddings: [[Float]],
        promptEmbeddings: [[Float]],
        spanIndex: [[Float]],
        spanMask: [[Float]],
        labelCount: Int,
        maxWidth: Int
    ) async throws -> [[[Float]]] {
        guard !wordEmbeddings.isEmpty else { return [] }
        guard !promptEmbeddings.isEmpty else { return [] }
        guard wordEmbeddings.count <= limits.maxWordCount else {
            throw GLiNERError.invalidInput("word count \(wordEmbeddings.count) exceeds scoring head limit of \(limits.maxWordCount)")
        }
        guard promptEmbeddings.count <= limits.maxPromptCount else {
            throw GLiNERError.invalidInput("label count \(promptEmbeddings.count) exceeds scoring head limit of \(limits.maxPromptCount)")
        }
        let spanCount = spanIndex.count
        guard spanCount == wordEmbeddings.count * maxWidth else {
            throw GLiNERError.invalidInput("Span index count (\(spanCount)) must equal wordCount * maxWidth (\(wordEmbeddings.count * maxWidth))")
        }
        let spanMaskColumns = spanMask.first?.count ?? 0
        guard spanMaskColumns == maxWidth else {
            throw GLiNERError.invalidInput("Span mask width mismatch: expected \(maxWidth), got \(spanMaskColumns)")
        }
        guard spanMask.count == wordEmbeddings.count else {
            throw GLiNERError.invalidInput("Span mask length mismatch: expected \(wordEmbeddings.count), got \(spanMask.count)")
        }
        let hiddenDim = wordEmbeddings[0].count
        guard hiddenDim > 0 else { return [] }
        let labelDim = promptEmbeddings[0].count
        guard labelDim == hiddenDim else {
            throw GLiNERError.invalidInput("Prompt embedding width (\(labelDim)) must match hidden dimension (\(hiddenDim))")
        }
        let wordArray = try createMLMultiArray(from: wordEmbeddings, leadingShape: [1, wordEmbeddings.count, hiddenDim])
        let promptArray = try createMLMultiArray(from: promptEmbeddings, leadingShape: [1, promptEmbeddings.count, hiddenDim])
        let spanIndexArray = try createMLMultiArray(from: spanIndex, leadingShape: [1, spanIndex.count, 2])
        let spanMaskArray = try createMLMultiArray(from: spanMask, leadingShape: [1, spanMask.count, maxWidth])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "word_embeddings": MLFeatureValue(multiArray: wordArray),
            "prompt_embeddings": MLFeatureValue(multiArray: promptArray),
            "span_idx": MLFeatureValue(multiArray: spanIndexArray),
            "span_mask": MLFeatureValue(multiArray: spanMaskArray)
        ])
        let prediction = try await withCheckedThrowingContinuation { continuation in
            predictionQueue.async {
                do {
                    let output = try self.model.prediction(from: provider)
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        guard let featureValue = prediction.featureValue(for: outputFeatureName), let scoresArray = featureValue.multiArrayValue else {
            let available = prediction.featureNames.joined(separator: ", ")
            throw GLiNERError.invalidOutput("Could not extract span scores from model output (expected \(outputFeatureName), available: [\(available)])")
        }
        return try extractScores(from: scoresArray, trimClassesTo: labelCount)
    }
    
    private func createMLMultiArray(from matrix: [[Float]], leadingShape: [Int]) throws -> MLMultiArray {
        let rowCount = matrix.count
        guard rowCount > 0 else {
            return try MLMultiArray(shape: leadingShape as [NSNumber], dataType: .float32)
        }
        let columnCount = matrix[0].count
        for row in matrix where row.count != columnCount {
            throw GLiNERError.invalidInput("Inconsistent inner dimension in matrix; expected \(columnCount)")
        }
        let elementCount = rowCount * columnCount
        let mlArray = try MLMultiArray(shape: leadingShape as [NSNumber], dataType: .float32)
        try mlArray.withUnsafeMutableFloat32Buffer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            precondition(buffer.count >= elementCount, "Unexpected MLMultiArray size: have \(buffer.count), need \(elementCount)")
            var writeIndex = 0
            for row in matrix {
                row.withUnsafeBufferPointer { rowBuffer in
                    guard let rowBase = rowBuffer.baseAddress else { return }
                    baseAddress.advanced(by: writeIndex).update(from: rowBase, count: columnCount)
                }
                writeIndex += columnCount
            }
        }
        return mlArray
    }
    
    private func extractScores(from multiArray: MLMultiArray, trimClassesTo labelCount: Int) throws -> [[[Float]]] {
        guard multiArray.shape.count == 4 else {
            throw GLiNERError.invalidOutput("Expected 4D tensor for span scores, received \(multiArray.shape.count)D tensor")
        }
        let batchSize = multiArray.shape[0].intValue
        let wordCount = multiArray.shape[1].intValue
        let widthCount = multiArray.shape[2].intValue
        let classCount = multiArray.shape[3].intValue
        guard batchSize == 1 else {
            throw GLiNERError.invalidOutput("Scoring head only supports batch size 1; received \(batchSize)")
        }
        let targetClassCount = min(labelCount, classCount)
        var result = Array(repeating: Array(repeating: [Float](repeating: -Float.greatestFiniteMagnitude, count: targetClassCount), count: widthCount), count: wordCount)
        for wordIndex in 0..<wordCount {
            for widthIndex in 0..<widthCount {
                for classIndex in 0..<targetClassCount {
                    let index = [0, wordIndex, widthIndex, classIndex] as [NSNumber]
                    let value = multiArray[index].floatValue
                    result[wordIndex][widthIndex][classIndex] = value
                }
            }
        }
        return result
    }

    private static func compileIfNeeded(at url: URL) throws -> URL {
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
    
    private static func resolveOutputFeatureName(for model: MLModel) -> String {
        let outputs = model.modelDescription.outputDescriptionsByName
        if let firstKey = outputs.keys.first {
            return firstKey
        }
        return "scores"
    }
}

@available(macOS 10.13, iOS 11.0, tvOS 11.0, watchOS 4.0, *)
extension GLiNERSpanScoringModel: @unchecked Sendable {}
