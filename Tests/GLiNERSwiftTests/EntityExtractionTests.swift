import XCTest
@testable import GLiNERSwift

/// Tests for entity extraction parity with Python implementation
final class EntityExtractionTests: XCTestCase {
    var fixtures: TestFixtures?
    var model: GLiNERModel?
    
    override func setUp() async throws {
        fixtures = try TestFixtures.load()
        let manifest = TestPaths.glinerManifestURL(file: #filePath)
        model = try await GLiNERModel(manifestURL: manifest)
    }
    
    func testEntityExtraction() async throws {
        guard let fixtures = fixtures, let model = model else {
            XCTFail("Fixtures or model not loaded")
            return
        }
        
        let scoreTolerance = fixtures.metadata.tolerance.entityScores
        
        for testCase in fixtures.testCases {
            let entities = try await model.extractEntities(
                from: testCase.text,
                labels: testCase.labels,
                threshold: testCase.threshold
            )
            if testCase.id == "simple_person" {
                print("[DEBUG] simple_person actual entities: \(entities)")
            }
            if testCase.id == "multi_entity" {
                print("[DEBUG] multi_entity actual entities: \(entities)")
            }
            
            // Sort both arrays by text for comparison
            let actualSorted = entities.sorted { $0.text < $1.text }
            let expectedSorted = testCase.entities.sorted { $0.text < $1.text }
            
            // Check entity count
            XCTAssertEqual(
                actualSorted.count,
                expectedSorted.count,
                "Entity count mismatch for test case: \(testCase.id)"
            )
            
            // Check each entity
            for (actual, expected) in zip(actualSorted, expectedSorted) {
                XCTAssertEqual(
                    actual.text.lowercased(),
                    expected.text.lowercased(),
                    "Entity text mismatch for test case: \(testCase.id)"
                )
                
                XCTAssertEqual(
                    actual.label,
                    expected.label,
                    "Entity label mismatch for test case: \(testCase.id)"
                )
                
                // Check score within tolerance
                let scoreDiff = abs(actual.score - expected.score)
                XCTAssertLessThanOrEqual(
                    scoreDiff,
                    scoreTolerance,
                    "Entity score mismatch for '\(actual.text)' in test case: \(testCase.id)"
                )
            }
        }
    }
    
    func testEmptyInput() async throws {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        let entities = try await model.extractEntities(
            from: "",
            labels: ["person", "organization"]
        )
        
        XCTAssertTrue(entities.isEmpty, "Empty input should return no entities")
    }
    
    func testNoLabels() async throws {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        let entities = try await model.extractEntities(
            from: "John Smith works at Apple.",
            labels: []
        )
        
        XCTAssertTrue(entities.isEmpty, "No labels should return no entities")
    }
    
    func testHighThreshold() async throws {
        guard let model = model else {
            XCTFail("Model not loaded")
            return
        }
        
        let entities = try await model.extractEntities(
            from: "John Smith works at Apple.",
            labels: ["person", "organization"],
            threshold: 0.99
        )
        
        // Very high threshold should return few or no entities
        XCTAssertLessThanOrEqual(entities.count, 2)
    }
}
