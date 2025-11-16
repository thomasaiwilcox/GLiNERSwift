import XCTest
@testable import GLiNERSwift

final class GLiNER2SchemaEncodingTests: XCTestCase {
    func testSchemaEncodingProducesMappings() throws {
        let tokenizerURL = TestPaths.projectRoot(file: #filePath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("tokenizer")
        let tokenizer = try GLiNERTokenizer(
            maxLength: 256,
            tokenizerDirectory: tokenizerURL,
            gliner2Config: GLiNER2PromptConfiguration()
        )
        let manifestURL = TestPaths.glinerManifestURL(file: #filePath)
        let manifest = try GLiNERManifest.load(from: manifestURL)

        let labels = ["company", "person", "location"]
        let encoding = try tokenizer.encodeGLiNER2SchemaInput(
            text: "ACME Corp hired Jane Doe in Paris.",
            labels: labels,
            maxSpanWidth: manifest.maxWidth
        )

        XCTAssertEqual(encoding.schemaTokensList.count, 1)
        XCTAssertEqual(encoding.entityLabels, labels)
        XCTAssertEqual(encoding.textTokens.count, encoding.textWordRanges.count)
        XCTAssertEqual(encoding.inputIds.count, encoding.attentionMask.count)
        XCTAssertEqual(encoding.inputIds.count, encoding.mappedIndices.count)
        XCTAssertTrue(encoding.mappedIndices.contains { $0.segment == .schema })
        XCTAssertTrue(encoding.mappedIndices.contains { $0.segment == .text })
        XCTAssertEqual(encoding.maxSpanWidth, manifest.maxWidth)
        XCTAssertEqual(encoding.spanMask.count, encoding.textTokens.count)
        XCTAssertEqual(encoding.spanIndices.count, encoding.textTokens.count * encoding.maxSpanWidth)
        XCTAssertTrue(encoding.promptLocations.contains { $0.kind == .prompt })
        let entityMarkers = encoding.promptLocations.filter { $0.kind == .entity }
        XCTAssertEqual(entityMarkers.count, labels.count)
        XCTAssertGreaterThanOrEqual(encoding.sequenceCapacity, encoding.textTokens.count)
    }
}
