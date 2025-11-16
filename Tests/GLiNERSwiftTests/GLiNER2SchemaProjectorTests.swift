import XCTest
@testable import GLiNERSwift

final class GLiNER2SchemaProjectorTests: XCTestCase {
    func testProjectorBuildsWordAndPromptEmbeddings() throws {
        let tokenizerURL = TestPaths.projectRoot(file: #filePath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("tokenizer")
        let tokenizer = try GLiNERTokenizer(
            maxLength: 256,
            tokenizerDirectory: tokenizerURL,
            gliner2Config: GLiNER2PromptConfiguration()
        )
        let manifest = try GLiNERManifest.load(from: TestPaths.glinerManifestURL(file: #filePath))
        let encoding = try tokenizer.encodeGLiNER2SchemaInput(
            text: "ACME Corp hired Jane Doe in Paris.",
            labels: ["company", "person", "location"],
            maxSpanWidth: manifest.maxWidth
        )

        let hiddenStates = makeSequentialHiddenStates(count: encoding.mappedIndices.count, width: 2)
        let projection = try GLiNER2SchemaProjector.project(hiddenStates: hiddenStates, encoding: encoding)

        XCTAssertEqual(projection.wordEmbeddings.count, encoding.textTokens.count)
        XCTAssertEqual(projection.schemaPromptEmbeddings.count, encoding.schemaTokensList.count)
        XCTAssertFalse(projection.schemaPromptEmbeddings[0].isEmpty)

        // Validate that each word embedding equals the first subword (not average).
        let textStart = encoding.textStartOriginalIndex
        for (wordIndex, wordEmbedding) in projection.wordEmbeddings.enumerated() {
            let firstSubwordIndex = encoding.mappedIndices.enumerated().first { offset, mapping in
                guard mapping.segment == .text else { return false }
                let mappedWord = mapping.originalIndex - textStart
                return mappedWord == wordIndex
            }
            guard let (offset, _) = firstSubwordIndex else {
                XCTFail("No subword found for word index \(wordIndex)")
                continue
            }
            let expected = hiddenStates[offset]
            XCTAssertEqual(wordEmbedding[0], expected[0], accuracy: 1e-5)
            XCTAssertEqual(wordEmbedding[1], expected[1], accuracy: 1e-5)
        }

        // Ensure we captured prompt embeddings (at least the [P] token).
        let firstSchemaPrompts = projection.schemaPromptEmbeddings[0]
        XCTAssertTrue(firstSchemaPrompts.contains { $0.kind == .prompt })
        let promptEmbedding = firstSchemaPrompts.first { $0.kind == .prompt }
        let promptLocation = encoding.promptLocations.first { $0.kind == .prompt }
        XCTAssertNotNil(promptEmbedding)
        XCTAssertNotNil(promptLocation)
        if let promptEmbedding, let promptLocation {
            let indices = Array(promptLocation.subwordRange)
            let expected = averageHiddenStates(indices: indices, source: hiddenStates)
            XCTAssertEqual(promptEmbedding.vector[0], expected[0], accuracy: 1e-5)
            XCTAssertEqual(promptEmbedding.vector[1], expected[1], accuracy: 1e-5)
        }
    }

    func testProjectorThrowsOnMismatchedHiddenStateCount() throws {
        let tokenizerURL = TestPaths.projectRoot(file: #filePath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("tokenizer")
        let tokenizer = try GLiNERTokenizer(
            maxLength: 128,
            tokenizerDirectory: tokenizerURL,
            gliner2Config: GLiNER2PromptConfiguration()
        )
        let manifest = try GLiNERManifest.load(from: TestPaths.glinerManifestURL(file: #filePath))
        let encoding = try tokenizer.encodeGLiNER2SchemaInput(
            text: "John Smith",
            labels: ["person"],
            maxSpanWidth: manifest.maxWidth
        )
        let hiddenStates = makeSequentialHiddenStates(count: max(1, encoding.mappedIndices.count - 1), width: 2)
        XCTAssertThrowsError(try GLiNER2SchemaProjector.project(hiddenStates: hiddenStates, encoding: encoding)) { error in
            guard case GLiNERError.encodingError = error else {
                return XCTFail("Expected encoding error, got: \(error)")
            }
        }
    }
}

private func makeSequentialHiddenStates(count: Int, width: Int) -> [[Float]] {
    return (0..<count).map { index in
        (0..<width).map { dim in Float(index * width + dim) }
    }
}

private func averageHiddenStates(indices: [Int], source: [[Float]]) -> [Float] {
    guard let firstIndex = indices.first else { return [] }
    var accumulator = source[firstIndex]
    if indices.count == 1 {
        return accumulator
    }
    for idx in indices.dropFirst() {
        let vector = source[idx]
        for dim in 0..<accumulator.count {
            accumulator[dim] += vector[dim]
        }
    }
    let scale = 1 / Float(indices.count)
    for dim in 0..<accumulator.count {
        accumulator[dim] *= scale
    }
    return accumulator
}
