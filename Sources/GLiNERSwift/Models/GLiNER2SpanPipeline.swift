import Foundation
@preconcurrency import CoreML

/// Coordinates the GLiNER2 span, classifier, and count predictor heads for entity extraction tasks.
struct GLiNER2SpanPipeline {
    struct Result {
        let spanEmbeddings: [[[Float]]]
        let spanMask: [[Float]]
        let wordEmbeddings: [[Float]]
        let labelEmbeddings: [[Float]]
        let promptEmbeddings: [[Float]]
        let classifierLogits: MLMultiArray
        let countLogits: MLMultiArray
        let structureEmbeddings: [[[Float]]]
        let predictedCount: Int
        let labels: [String]
    }

    private let spanRepresentation: GLiNER2SpanRepresentationModel
    private let classifier: GLiNER2ClassifierModel
    private let countPredictor: GLiNER2CountPredictorModel
    private let countEmbed: GLiNER2CountEmbedModel
    private let sequenceCapacity: Int
    private let maxCount: Int

    init(
        spanRepresentation: GLiNER2SpanRepresentationModel,
        classifier: GLiNER2ClassifierModel,
        countPredictor: GLiNER2CountPredictorModel,
        countEmbed: GLiNER2CountEmbedModel,
        sequenceCapacity: Int,
        maxCount: Int
    ) {
        self.spanRepresentation = spanRepresentation
        self.classifier = classifier
        self.countPredictor = countPredictor
        self.countEmbed = countEmbed
        self.sequenceCapacity = sequenceCapacity
        self.maxCount = maxCount
    }

    init(resources: GLiNER2Resources, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let spanModel = try GLiNER2SpanRepresentationModel(modelURL: resources.spanRepresentationURL, computeUnits: computeUnits)
        let classifierModel = try GLiNER2ClassifierModel(modelURL: resources.classifierURL, computeUnits: computeUnits)
        let countModel = try GLiNER2CountPredictorModel(modelURL: resources.countPredictorURL, computeUnits: computeUnits)
        let embedModel = try GLiNER2CountEmbedModel(
            modelURL: resources.countEmbedURL,
            schemaTokenCapacity: resources.manifest.maxSchemaTokens,
            maxCount: resources.manifest.maxCount,
            computeUnits: computeUnits
        )
        self.init(
            spanRepresentation: spanModel,
            classifier: classifierModel,
            countPredictor: countModel,
            countEmbed: embedModel,
            sequenceCapacity: resources.manifest.maxSeqLen,
            maxCount: resources.manifest.maxCount
        )
    }

    /// Runs the GLiNER2 span pipeline for an entity schema, returning the raw Core ML outputs needed for decoding spans.
    func run(hiddenStates: [[Float]], encoding: GLiNER2SchemaEncoding) async throws -> Result {
        guard !encoding.entityLabels.isEmpty else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding must include at least one entity label")
        }
        guard !encoding.spanIndices.isEmpty else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding is missing span indices")
        }
        guard !encoding.spanMask.isEmpty else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding is missing span mask")
        }

        let projection = try GLiNER2SchemaProjector.project(hiddenStates: hiddenStates, encoding: encoding)
        guard let schemaPrompts = projection.schemaPromptEmbeddings.first, !schemaPrompts.isEmpty else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding did not produce schema prompt embeddings")
        }
        guard let promptEmbedding = schemaPrompts.first(where: { $0.kind == .prompt })?.vector else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding is missing a [P] prompt embedding")
        }
        let specialEmbeddings = schemaPrompts.map { $0.vector }
        let labelEmbeddings = schemaPrompts
            .filter { $0.kind == .entity }
            .map { $0.vector }
        guard labelEmbeddings.count == encoding.entityLabels.count else {
            throw GLiNERError.encodingError("Entity prompt count (\(labelEmbeddings.count)) does not match label list (\(encoding.entityLabels.count))")
        }
        if encoding.textTokens.count < 40,
           let firstLabel = labelEmbeddings.first {
            let labelNorm = sqrt(firstLabel.reduce(0) { $0 + $1 * $1 })
            let labelSample = firstLabel.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", ")
            print("[DEBUG] Schema label norm: \(labelNorm) sample: [\(labelSample)]")
        }

        guard let hiddenSize = projection.wordEmbeddings.first?.count else {
            throw GLiNERError.encodingError("GLiNER2 schema encoding did not produce word-level embeddings")
        }
        let paddedEmbeddings = padTokenEmbeddings(
            projection.wordEmbeddings,
            capacity: sequenceCapacity,
            hiddenSize: hiddenSize
        )
        let paddedSpanIndices = padSpanIndices(
            encoding.spanIndices,
            sequenceLength: sequenceCapacity,
            maxWidth: encoding.maxSpanWidth
        )

        let spanEmbeddings = try await spanRepresentation.spanEmbeddings(
            tokenEmbeddings: paddedEmbeddings,
            spanIndices: paddedSpanIndices
        )
        let trimmedSpanEmbeddings = Array(spanEmbeddings.prefix(encoding.textTokens.count))
        let classifierLogits = try await classifier.logits(schemaEmbeddings: specialEmbeddings)
        let countLogits = try await countPredictor.predict(promptEmbeddings: [promptEmbedding])
        let predictedCount = GLiNER2SpanPipeline.argmaxCount(from: countLogits, maxCount: maxCount)
        let structureEmbeddings: [[[Float]]]
        if predictedCount > 0 {
            let projections = try await countEmbed.projectedEmbeddings(labelEmbeddings: labelEmbeddings)
            structureEmbeddings = Array(projections.prefix(predictedCount))
        } else {
            structureEmbeddings = []
        }
        if encoding.textTokens.count < 40,
           let firstInstance = structureEmbeddings.first?.first {
            let sample = firstInstance.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", ")
            print("[DEBUG] Structure embedding sample: [\(sample)]")
        }

        return Result(
            spanEmbeddings: trimmedSpanEmbeddings,
            spanMask: encoding.spanMask,
            wordEmbeddings: projection.wordEmbeddings,
            labelEmbeddings: labelEmbeddings,
            promptEmbeddings: specialEmbeddings,
            classifierLogits: classifierLogits,
            countLogits: countLogits,
            structureEmbeddings: structureEmbeddings,
            predictedCount: predictedCount,
            labels: encoding.entityLabels
        )
    }
}

private extension GLiNER2SpanPipeline {
    func padTokenEmbeddings(_ embeddings: [[Float]], capacity: Int, hiddenSize: Int) -> [[Float]] {
        guard embeddings.count < capacity else {
            return Array(embeddings.prefix(capacity))
        }
        var padded = embeddings
        padded.reserveCapacity(capacity)
        let zeros = [Float](repeating: 0, count: hiddenSize)
        while padded.count < capacity {
            padded.append(zeros)
        }
        return padded
    }

    func padSpanIndices(_ spanIndices: [[Int32]], sequenceLength: Int, maxWidth: Int) -> [[Int32]] {
        let requiredCount = sequenceLength * maxWidth
        if spanIndices.count >= requiredCount {
            return Array(spanIndices.prefix(requiredCount))
        }
        var padded = spanIndices
        padded.reserveCapacity(requiredCount)
        let paddingPair: [Int32] = [0, 0]
        while padded.count < requiredCount {
            padded.append(paddingPair)
        }
        return padded
    }

    static func argmaxCount(from logits: MLMultiArray, maxCount: Int) -> Int {
        do {
            let values: [Float]
            switch logits.dataType {
            case .float32:
                values = try logits.withUnsafeFloat32Buffer { Array($0) }
            case .float16:
                values = try logits.withUnsafeFloat16Buffer { buffer in
                    guard let base = buffer.baseAddress else { return [] }
                    return (0..<buffer.count).map { Float(base[$0]) }
                }
            default:
                return 0
            }
            guard !values.isEmpty else { return 0 }
            var bestIndex = 0
            var bestValue = values[0]
            for (idx, value) in values.enumerated() where value > bestValue {
                bestValue = value
                bestIndex = idx
            }
            return min(bestIndex, maxCount)
        } catch {
            return 0
        }
    }
}
