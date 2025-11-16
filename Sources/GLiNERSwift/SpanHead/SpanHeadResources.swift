import Foundation

/// Loads the exported span-head projection weights from bundled resources.
final class SpanHeadResources {
    let metadata: SpanHeadMetadata
    let projectStart: FeedForwardLayer
    let projectEnd: FeedForwardLayer
    let outProject: FeedForwardLayer
    let promptProjection: FeedForwardLayer
    let rnn: BiLSTM

    init(bundle: Bundle = .module) throws {
        guard let baseURL = bundle.resourceURL?.appendingPathComponent("SpanHead", isDirectory: true) else {
            throw GLiNERError.encodingError("SpanHead resources directory is missing from the bundle")
        }

        metadata = try SpanHeadMetadataProvider.metadata(bundle: bundle)

        func loadLinear(_ info: SpanHeadMetadata.LinearInfo) throws -> LinearLayer {
            let weight = try SpanHeadResources.loadTensor(at: baseURL, fileName: info.weight, count: info.outFeatures * info.inFeatures)
            let bias = try SpanHeadResources.loadTensor(at: baseURL, fileName: info.bias, count: info.outFeatures)
            return LinearLayer(weights: weight, bias: bias, inFeatures: info.inFeatures, outFeatures: info.outFeatures)
        }

        func loadProjection(_ info: SpanHeadMetadata.ProjectionInfo) throws -> FeedForwardLayer {
            let fc1 = try loadLinear(info.fc1)
            let fc2 = try loadLinear(info.fc2)
            return FeedForwardLayer(first: fc1, second: fc2)
        }

        projectStart = try loadProjection(metadata.layers.projectStart)
        projectEnd = try loadProjection(metadata.layers.projectEnd)
        outProject = try loadProjection(metadata.layers.outProject)
        promptProjection = try loadProjection(metadata.layers.promptProjection)

        let rnnInfo = metadata.rnn
        let forwardLSTM = try SpanHeadResources.loadLSTMWeights(at: baseURL, info: rnnInfo.forward, inputSize: rnnInfo.inputSize, hiddenSize: rnnInfo.hiddenSize)
        let backwardLSTM = try rnnInfo.backward.map { try SpanHeadResources.loadLSTMWeights(at: baseURL, info: $0, inputSize: rnnInfo.inputSize, hiddenSize: rnnInfo.hiddenSize) }
        rnn = BiLSTM(
            inputSize: rnnInfo.inputSize,
            hiddenSize: rnnInfo.hiddenSize,
            forward: forwardLSTM,
            backward: backwardLSTM
        )
    }

    private static func loadTensor(at baseURL: URL, fileName: String, count: Int) throws -> [Float] {
        let url = baseURL.appendingPathComponent(fileName)
        let data = try Data(contentsOf: url)
        let expectedBytes = count * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            throw GLiNERError.encodingError("Unexpected tensor byte count for \(fileName): expected \(expectedBytes), got \(data.count)")
        }

        return data.withUnsafeBytes { rawBuffer -> [Float] in
            let buffer = rawBuffer.bindMemory(to: Float.self)
            return Array(buffer)
        }
    }

    private static func loadLSTMWeights(at baseURL: URL, info: SpanHeadMetadata.LSTMWeights, inputSize: Int, hiddenSize: Int) throws -> LSTMWeightsSet {
        let gateCount = hiddenSize * 4
        let weightIH = try loadTensor(at: baseURL, fileName: info.weightIH, count: gateCount * inputSize)
        let weightHH = try loadTensor(at: baseURL, fileName: info.weightHH, count: gateCount * hiddenSize)
        let bias = try loadTensor(at: baseURL, fileName: info.bias, count: gateCount)
        return LSTMWeightsSet(weightIH: weightIH, weightHH: weightHH, bias: bias)
    }
}
