import XCTest
@testable import GLiNERSwift

final class GLiNER2SpanPipelineTests: XCTestCase {
    func testPipelineRunsEndToEnd() async throws {
        let manifestURL = TestPaths.glinerManifestURL(file: #filePath)
        let resources = try GLiNER2Resources(manifestURL: manifestURL)

        let tokenizer = try GLiNERTokenizer(
            maxLength: 256,
            tokenizerDirectory: resources.tokenizerURL,
            gliner2Config: GLiNER2PromptConfiguration()
        )
        let labels = ["company", "person", "location"]
        let encoding = try tokenizer.encodeGLiNER2SchemaInput(
            text: "ACME Corp hired Jane Doe in Paris.",
            labels: labels,
            maxSpanWidth: resources.manifest.maxWidth
        )

        let encoder = try GLiNEREncoder(modelURL: resources.encoderURL)
        let hiddenStates = try await encoder.encode(
            inputIds: encoding.inputIds,
            attentionMask: encoding.attentionMask
        )

        let pipeline = try GLiNER2SpanPipeline(resources: resources)
        let result = try await pipeline.run(hiddenStates: hiddenStates, encoding: encoding)

        XCTAssertEqual(result.labels, labels)
        XCTAssertEqual(result.labelEmbeddings.count, labels.count)
        XCTAssertEqual(result.spanEmbeddings.count, encoding.textTokens.count)
        XCTAssertEqual(result.spanMask.count, encoding.spanMask.count)
        XCTAssertEqual(result.wordEmbeddings.count, encoding.textTokens.count)
        XCTAssertGreaterThan(result.promptEmbeddings.count, 0)
        XCTAssertEqual(result.structureEmbeddings.count, result.predictedCount)
        XCTAssertGreaterThan(result.predictedCount, 0)
    }
}
