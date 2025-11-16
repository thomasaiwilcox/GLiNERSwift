import XCTest
@testable import GLiNERSwift

/// Tests for the Schema builder API and GLiNER2 convenience methods
final class GLiNER2SchemaAPITests: XCTestCase {
    
    func testSchemaBuilderEntities() throws {
        // Test basic entity schema building
        let schema = Schema()
            .entities(["person", "organization", "location"])
        
        let schemaDict = schema.build()
        
        // Verify entities are in the schema
        XCTAssertNotNil(schemaDict["entities"])
        let entities = schemaDict["entities"] as? [[String: Any]]
        XCTAssertEqual(entities?.count, 3)
        
        // Verify entity order is preserved
        let firstEntity = entities?[0]["entity"] as? String
        XCTAssertEqual(firstEntity, "person")
    }
    
    func testSchemaBuilderClassification() throws {
        let schema = Schema()
            .classification(
                "sentiment",
                labels: ["positive", "negative", "neutral"],
                multiLabel: false,
                clsThreshold: 0.7
            )
        
        let schemaDict = schema.build()
        
        XCTAssertNotNil(schemaDict["classifications"])
        let classifications = schemaDict["classifications"] as? [[String: Any]]
        XCTAssertEqual(classifications?.count, 1)
        
        let firstTask = classifications?[0]
        XCTAssertEqual(firstTask?["task"] as? String, "sentiment")
        XCTAssertEqual((firstTask?["labels"] as? [String])?.count, 3)
        XCTAssertEqual(firstTask?["multi_label"] as? Bool, false)
        XCTAssertEqual(firstTask?["cls_threshold"] as? Float, 0.7)
    }
    
    func testSchemaBuilderStructure() throws {
        let schema = Schema()
            .structure("person_info")
            .field("name", dtype: "str", description: "Person's name")
            .field("age", dtype: "str", description: "Person's age")
            .field("occupation", dtype: "list", description: "Jobs")
            .build()
        
        let schemaDict = schema.build()
        
        XCTAssertNotNil(schemaDict["json_structures"])
        let structures = schemaDict["json_structures"] as? [[String: [String: Any]]]
        XCTAssertEqual(structures?.count, 1)
        
        let firstStructure = structures?[0]
        XCTAssertNotNil(firstStructure?["person_info"])
        
        let fields = firstStructure?["person_info"]
        XCTAssertEqual(fields?.count, 3)
        XCTAssertNotNil(fields?["name"])
        XCTAssertNotNil(fields?["age"])
        XCTAssertNotNil(fields?["occupation"])
    }
    
    func testSchemaBuilderCombined() throws {
        // Test combining multiple schema types
        let schema = Schema()
            .entities(["person", "organization"])
            .classification("sentiment", labels: ["positive", "negative"])
            .structure("contact")
            .field("email", dtype: "str")
            .field("phone", dtype: "str")
            .build()
        
        let schemaDict = schema.build()
        
        XCTAssertNotNil(schemaDict["entities"])
        XCTAssertNotNil(schemaDict["classifications"])
        XCTAssertNotNil(schemaDict["json_structures"])
        
        let entities = schemaDict["entities"] as? [[String: Any]]
        XCTAssertEqual(entities?.count, 2)
        
        let classifications = schemaDict["classifications"] as? [[String: Any]]
        XCTAssertEqual(classifications?.count, 1)
        
        let structures = schemaDict["json_structures"] as? [[String: [String: Any]]]
        XCTAssertEqual(structures?.count, 1)
    }
    
    func testRegexValidator() throws {
        // Test full match
        let fullValidator = try RegexValidator(pattern: "^[0-9]{3}-[0-9]{4}$", mode: .full)
        XCTAssertTrue(fullValidator.validate("123-4567"))
        XCTAssertFalse(fullValidator.validate("123-456"))
        XCTAssertFalse(fullValidator.validate("abc-defg"))
        
        // Test partial match
        let partialValidator = try RegexValidator(pattern: "[0-9]+", mode: .partial)
        XCTAssertTrue(partialValidator.validate("abc123def"))
        XCTAssertTrue(partialValidator.validate("123"))
        XCTAssertFalse(partialValidator.validate("abcdef"))
        
        // Test exclude mode
        let excludeValidator = try RegexValidator(pattern: "[0-9]+", mode: .partial, exclude: true)
        XCTAssertFalse(excludeValidator.validate("abc123def"))
        XCTAssertTrue(excludeValidator.validate("abcdef"))
    }
    
    func testEntityTypesInputArray() throws {
        let schema = Schema()
            .entities(.array(["person", "place", "thing"]))
        
        let schemaDict = schema.build()
        let entities = schemaDict["entities"] as? [[String: Any]]
        XCTAssertEqual(entities?.count, 3)
    }
    
    func testEntityTypesInputDictionary() throws {
        let schema = Schema()
            .entities(.dictionary([
                "person": .string("A person's name"),
                "organization": .dictionary(["description": "A company or org"])
            ]))
        
        let schemaDict = schema.build()
        let entities = schemaDict["entities"] as? [[String: Any]]
        XCTAssertEqual(entities?.count, 2)
        
        // Verify descriptions are stored
        let firstEntity = entities?.first { ($0["entity"] as? String) == "person" }
        XCTAssertEqual(firstEntity?["description"] as? String, "A person's name")
    }
    
    func testStructureBuilderAutoFinish() throws {
        // Test that structure auto-finishes when builder goes out of scope
        let schema = Schema()
        
        do {
            let _ = schema
                .structure("test")
                .field("field1")
                .field("field2")
            // Builder should auto-finish here when it goes out of scope
        }
        
        let schemaDict = schema.build()
        let structures = schemaDict["json_structures"] as? [[String: [String: Any]]]
        XCTAssertEqual(structures?.count, 1)
    }
    
    func testFieldMetadataStorage() throws {
        let schema = Schema()
            .structure("person")
            .field("name", dtype: "str", threshold: 0.8)
            .field("emails", dtype: "list", threshold: 0.6)
            .build()
        
        // Verify metadata is stored
        let nameMetadata = schema.fieldMetadata["person.name"]
        XCTAssertNotNil(nameMetadata)
        XCTAssertEqual(nameMetadata?.dtype, "str")
        XCTAssertEqual(nameMetadata?.threshold, 0.8)
        
        let emailsMetadata = schema.fieldMetadata["person.emails"]
        XCTAssertEqual(emailsMetadata?.dtype, "list")
        XCTAssertEqual(emailsMetadata?.threshold, 0.6)
    }
    
    func testFieldOrderPreservation() throws {
        let schema = Schema()
            .structure("ordered")
            .field("third")
            .field("first")
            .field("second")
            .build()
        
        let order = schema.fieldOrders["ordered"]
        XCTAssertEqual(order, ["third", "first", "second"])
    }
    
    func testEntityOrderPreservation() throws {
        let schema = Schema()
            .entities(["zebra", "apple", "monkey"])
        
        XCTAssertEqual(schema.entityOrder, ["zebra", "apple", "monkey"])
    }
}
