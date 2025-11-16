import XCTest
@testable import GLiNERSwift

final class GLiNER2ResourcesTests: XCTestCase {
    func testLoadsManifestAndResolvesAllArtifacts() throws {
        let manifest = TestPaths.projectRoot(file: #filePath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("export_manifest.json")
        let resources = try GLiNER2Resources(manifestURL: manifest)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.encoderURL.path), resources.encoderURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.spanRepresentationURL.path), resources.spanRepresentationURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.classifierURL.path), resources.classifierURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.countPredictorURL.path), resources.countPredictorURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.countEmbedURL.path), resources.countEmbedURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resources.tokenizerURL.path), resources.tokenizerURL.path)
        XCTAssertEqual(resources.manifest.maxSeqLen, 512)
        XCTAssertEqual(resources.manifest.maxWidth, 8)
    }
}
