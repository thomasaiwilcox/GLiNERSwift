import XCTest
@testable import GLiNERSwift

final class GLiNER2CoreMLHeadTests: XCTestCase {
    private var resources: GLiNER2Resources!

    override func setUpWithError() throws {
        let manifest = TestPaths.glinerManifestURL(file: #filePath)
        resources = try GLiNER2Resources(manifestURL: manifest)
    }

    override func tearDownWithError() throws {
        resources = nil
    }

    func testSpanRepresentationProducesEmbeddings() async throws {
        let spanModel = try GLiNER2SpanRepresentationModel(modelURL: resources.spanRepresentationURL)
        let manifest = resources.manifest
        let sequenceLength = manifest.maxSeqLen
        let hiddenSize = manifest.hiddenSize
        let maxWidth = manifest.maxWidth

        var tokenEmbeddings = Array(
            repeating: Array(repeating: Float(0), count: hiddenSize),
            count: sequenceLength
        )
        // Write a simple ramp so the input tensor is non-constant.
        var value: Float = 0
        for seqIdx in 0..<sequenceLength {
            for hiddenIdx in 0..<hiddenSize {
                tokenEmbeddings[seqIdx][hiddenIdx] = value
                value += 0.0001
            }
        }

        var spans: [[Int32]] = []
        spans.reserveCapacity(sequenceLength * maxWidth)
        for start in 0..<sequenceLength {
            for width in 0..<maxWidth {
                let end = min(sequenceLength - 1, start + width)
                spans.append([Int32(start), Int32(end)])
            }
        }

        let embeddings = try await spanModel.spanEmbeddings(
            tokenEmbeddings: tokenEmbeddings,
            spanIndices: spans
        )
        XCTAssertEqual(embeddings.count, sequenceLength)
        XCTAssertEqual(embeddings.first?.count, maxWidth)
        XCTAssertEqual(embeddings.first?.first?.count, hiddenSize)
    }

    func testClassifierRunsInference() async throws {
        let classifier = try GLiNER2ClassifierModel(modelURL: resources.classifierURL)
        let manifest = resources.manifest
        let promptCount = manifest.maxSchemaTokens
        let hiddenSize = manifest.hiddenSize
        var embeddings = Array(
            repeating: Array(repeating: Float(0.5), count: hiddenSize),
            count: promptCount
        )
        // Make the tensor non-uniform.
        for idx in 0..<promptCount {
            embeddings[idx][idx % hiddenSize] = Float(idx)
        }
        let logits = try await classifier.logits(schemaEmbeddings: embeddings)
        XCTAssertGreaterThan(logits.count, 0)
    }

    func testCountPredictorRunsInference() async throws {
        let predictor = try GLiNER2CountPredictorModel(modelURL: resources.countPredictorURL)
        let manifest = resources.manifest
        let promptCount = manifest.maxSchemaTokens
        let hiddenSize = manifest.hiddenSize
        let embeddings = Array(
            repeating: Array(repeating: Float(0.25), count: hiddenSize),
            count: promptCount
        )
        let counts = try await predictor.predict(promptEmbeddings: embeddings)
        XCTAssertGreaterThan(counts.count, 0)
    }

    func testCountEmbedRunsInference() async throws {
        let manifest = resources.manifest
        let embedder = try GLiNER2CountEmbedModel(
            modelURL: resources.countEmbedURL,
            schemaTokenCapacity: manifest.maxSchemaTokens,
            maxCount: manifest.maxCount
        )
        let hiddenSize = manifest.hiddenSize
        let labelCount = 4
        var embeddings = Array(
            repeating: Array(repeating: Float(0.1), count: hiddenSize),
            count: labelCount
        )
        for idx in 0..<labelCount {
            embeddings[idx][idx % hiddenSize] = Float(idx)
        }
        let projections = try await embedder.projectedEmbeddings(labelEmbeddings: embeddings)
        XCTAssertEqual(projections.count, manifest.maxCount)
        XCTAssertEqual(projections.first?.count, labelCount)
    }
}
