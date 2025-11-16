import XCTest
@testable import GLiNERSwift

/// Tests for encoder output parity with Python implementation
final class EncoderParityTests: XCTestCase {
    var fixtures: TestFixtures?
    var encoder: GLiNEREncoder?
    
    override func setUp() async throws {
        fixtures = try TestFixtures.load()
        encoder = try GLiNEREncoder()
    }
    
    func testEncoderOutputParity() async throws {
        guard let fixtures = fixtures, let encoder = encoder else {
            XCTFail("Fixtures or encoder not loaded")
            return
        }
        
        let tolerance: Float = fixtures.metadata.tolerance.hiddenStates
        
        for testCase in fixtures.testCases {
            guard let expectedHiddenStates = testCase.encoderOutput.hiddenStates else {
                continue
            }
            
            // Run encoder
            let hiddenStates = try await encoder.encode(
                inputIds: testCase.tokenizerOutput.inputIds,
                attentionMask: testCase.tokenizerOutput.attentionMask
            )
            
            // Check shape
            XCTAssertEqual(
                hiddenStates.count,
                expectedHiddenStates.count,
                "Sequence length mismatch for test case: \(testCase.id)"
            )
            
            guard hiddenStates.count > 0 else { continue }
            
            XCTAssertEqual(
                hiddenStates[0].count,
                expectedHiddenStates[0].count,
                "Hidden dimension mismatch for test case: \(testCase.id)"
            )
            
            // Check values within tolerance
            for (seqIdx, (actual, expected)) in zip(hiddenStates, expectedHiddenStates).enumerated() {
                for (dimIdx, (actualVal, expectedVal)) in zip(actual, expected).enumerated() {
                    let diff = abs(actualVal - expectedVal)
                    XCTAssertLessThanOrEqual(
                        diff,
                        tolerance,
                        "Hidden state mismatch at [\(seqIdx), \(dimIdx)] for test case: \(testCase.id)"
                    )
                }
            }
        }
    }
}
