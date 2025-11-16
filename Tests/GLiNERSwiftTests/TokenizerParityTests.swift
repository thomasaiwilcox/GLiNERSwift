import XCTest
@testable import GLiNERSwift

/// Tests for tokenizer parity with Python implementation
final class TokenizerParityTests: XCTestCase {
    var fixtures: TestFixtures?
    
    override func setUpWithError() throws {
        fixtures = try TestFixtures.load()
    }
    
    func testTokenizerParity() throws {
        let tokenizer = try GLiNERTokenizer()
        
        guard let fixtures = fixtures else {
            XCTFail("Fixtures not loaded")
            return
        }
        
        for testCase in fixtures.testCases {
            let tokenized = try tokenizer.encode(testCase.text)
            
            // Check token IDs match exactly
            XCTAssertEqual(
                tokenized.inputIds,
                testCase.tokenizerOutput.inputIds,
                "Token IDs mismatch for test case: \(testCase.id)"
            )
            
            // Check attention mask matches exactly
            XCTAssertEqual(
                tokenized.attentionMask,
                testCase.tokenizerOutput.attentionMask,
                "Attention mask mismatch for test case: \(testCase.id)"
            )
            
            // Check tokens match
            XCTAssertEqual(
                tokenized.tokens,
                testCase.tokenizerOutput.tokens,
                "Tokens mismatch for test case: \(testCase.id)"
            )
        }
    }

    func testTokenizerLoadsFromExternalArtifacts() throws {
        let externalTokenizerURL = TestPaths.projectRoot(file: #filePath)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("tokenizer")

        let tokenizer = try GLiNERTokenizer(maxLength: 128, tokenizerDirectory: externalTokenizerURL)
        let encoding = try tokenizer.encode("ACME Corp launches GLiNER2.")

        XCTAssertFalse(encoding.inputIds.isEmpty, "External tokenizer directory should produce token IDs")
        XCTAssertEqual(encoding.inputIds.count, encoding.attentionMask.count)
    }
}
