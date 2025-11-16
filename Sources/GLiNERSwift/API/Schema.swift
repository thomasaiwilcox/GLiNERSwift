import Foundation

/// Fluent API for building extraction schemas, matching GLiNER2 Python's Schema class.
public class Schema {
    private(set) var schemaDict: [String: Any] = [:]
    private(set) var fieldMetadata: [String: FieldMetadata] = [:]
    private(set) var entityMetadata: [String: EntityMetadata] = [:]
    private(set) var fieldOrders: [String: [String]] = [:]
    private(set) var entityOrder: [String] = []
    
    public struct EntityMetadata {
        let dtype: String
        let threshold: Float?
    }
    
    public struct FieldMetadata {
        let dtype: String
        let threshold: Float?
        let validators: [RegexValidator]?
    }
    
    public init() {}
    
    /// Add entity extraction to the schema.
    /// - Parameters:
    ///   - entityTypes: Entity types as array of strings
    ///   - dtype: Return type ("list" or "str")
    ///   - threshold: Optional per-entity threshold override
    /// - Returns: Self for chaining
    @discardableResult
    public func entities(
        _ entityTypes: [String],
        dtype: String = "list",
        threshold: Float? = nil
    ) -> Self {
        for entity in entityTypes {
            let entityDict: [String: Any] = [
                "entity": entity,
                "description": ""
            ]
            
            var entitiesList: [[String: Any]] = schemaDict["entities"] as? [[String: Any]] ?? []
            entitiesList.append(entityDict)
            schemaDict["entities"] = entitiesList
            
            // Store metadata
            entityMetadata[entity] = EntityMetadata(dtype: dtype, threshold: threshold)
            
            if !entityOrder.contains(entity) {
                entityOrder.append(entity)
            }
        }
        
        return self
    }
    
    /// Add entity extraction to the schema with detailed configuration.
    /// - Parameters:
    ///   - entityTypes: Entity types with configuration
    ///   - dtype: Return type ("list" or "str")
    ///   - threshold: Optional per-entity threshold override
    /// - Returns: Self for chaining
    @discardableResult
    public func entities(
        _ entityTypes: EntityTypesInput,
        dtype: String = "list",
        threshold: Float? = nil
    ) -> Self {
        let parsed = parseEntityInput(entityTypes)
        
        // Store in schema dict
        var entitiesList: [[String: Any]] = schemaDict["entities"] as? [[String: Any]] ?? []
        
        for (entity, config) in parsed {
            let entityDict: [String: Any] = [
                "entity": entity,
                "description": config["description"] as? String ?? ""
            ]
            entitiesList.append(entityDict)
            
            // Store metadata
            let entityDtype = config["dtype"] as? String ?? dtype
            let entityThreshold = config["threshold"] as? Float ?? threshold
            entityMetadata[entity] = EntityMetadata(dtype: entityDtype, threshold: entityThreshold)
            
            if !entityOrder.contains(entity) {
                entityOrder.append(entity)
            }
        }
        
        schemaDict["entities"] = entitiesList
        return self
    }
    
    /// Add text classification to the schema.
    /// - Parameters:
    ///   - task: Task name
    ///   - labels: Classification labels
    ///   - multiLabel: Whether to allow multiple labels
    ///   - clsThreshold: Confidence threshold for classification
    /// - Returns: Self for chaining
    @discardableResult
    public func classification(
        _ task: String,
        labels: [String],
        multiLabel: Bool = false,
        clsThreshold: Float = 0.5
    ) -> Self {
        var classificationsList: [[String: Any]] = schemaDict["classifications"] as? [[String: Any]] ?? []
        
        let classificationDict: [String: Any] = [
            "task": task,
            "labels": labels,
            "multi_label": multiLabel,
            "cls_threshold": clsThreshold,
            "true_label": ["N/A"]
        ]
        
        classificationsList.append(classificationDict)
        schemaDict["classifications"] = classificationsList
        
        return self
    }
    
    /// Start building a structured data extraction schema.
    /// - Parameter name: Structure name
    /// - Returns: StructureBuilder for adding fields
    public func structure(_ name: String) -> StructureBuilder {
        return StructureBuilder(schema: self, structureName: name)
    }
    
    /// Build the final schema dictionary.
    /// - Returns: Schema as dictionary
    public func build() -> [String: Any] {
        return schemaDict
    }
    
    // MARK: - Internal helpers
    
    func addStructure(_ name: String, fields: [String: Any]) {
        var structuresList: [[String: [String: Any]]] = schemaDict["json_structures"] as? [[String: [String: Any]]] ?? []
        structuresList.append([name: fields])
        schemaDict["json_structures"] = structuresList
    }
    
    func setFieldMetadata(_ key: String, metadata: FieldMetadata) {
        fieldMetadata[key] = metadata
    }
    
    func setFieldOrder(_ parent: String, order: [String]) {
        fieldOrders[parent] = order
    }
    
    private func parseEntityInput(_ input: EntityTypesInput) -> [String: [String: Any]] {
        var result: [String: [String: Any]] = [:]
        
        switch input {
        case .array(let entities):
            for entity in entities {
                result[entity] = [:]
            }
        case .dictionary(let dict):
            for (entity, config) in dict {
                switch config {
                case .string(let desc):
                    result[entity] = ["description": desc]
                case .dictionary(let configDict):
                    result[entity] = configDict
                }
            }
        }
        
        return result
    }
}

/// Builder for structured data extraction fields.
public class StructureBuilder {
    private let schema: Schema  // Strong reference to keep schema alive
    private let structureName: String
    private var fields: [String: Any] = [:]
    private var fieldOrder: [String] = []
    private var autoFinished = false
    
    init(schema: Schema, structureName: String) {
        self.schema = schema
        self.structureName = structureName
    }
    
    /// Add a field to the structure.
    /// - Parameters:
    ///   - name: Field name
    ///   - dtype: Data type ("list" or "str")
    ///   - choices: Optional list of valid choices
    ///   - description: Optional field description
    ///   - threshold: Optional confidence threshold
    ///   - validators: Optional regex validators
    /// - Returns: Self for chaining
    @discardableResult
    public func field(
        _ name: String,
        dtype: String = "list",
        choices: [String]? = nil,
        description: String? = nil,
        threshold: Float? = nil,
        validators: [RegexValidator]? = nil
    ) -> Self {
        var fieldDict: [String: Any] = [:]
        
        if let choices = choices {
            fieldDict["choices"] = choices
            fieldDict["value"] = ""
        }
        
        if let description = description {
            fieldDict["description"] = description
        }
        
        fields[name] = fieldDict
        fieldOrder.append(name)
        
        // Store metadata
        let fieldKey = "\(structureName).\(name)"
        schema.setFieldMetadata(fieldKey, metadata: Schema.FieldMetadata(
            dtype: dtype,
            threshold: threshold,
            validators: validators
        ))
        
        return self
    }
    
    /// Finish building the structure and return to the parent schema.
    /// - Returns: Parent Schema for continued chaining
    @discardableResult
    public func build() -> Schema {
        finishIfNeeded()
        return schema
    }
    
    private func finishIfNeeded() {
        guard !autoFinished else { return }
        schema.addStructure(structureName, fields: fields)
        schema.setFieldOrder(structureName, order: fieldOrder)
        autoFinished = true
    }
    
    deinit {
        finishIfNeeded()
    }
}

/// Input types for entity configuration
public enum EntityTypesInput {
    case array([String])
    case dictionary([String: EntityConfig])
}

public enum EntityConfig {
    case string(String)
    case dictionary([String: Any])
}

/// Regex validator for span filtering, matching Python's RegexValidator
public struct RegexValidator {
    public enum Mode {
        case full
        case partial
    }
    
    let pattern: String
    let mode: Mode
    let exclude: Bool
    private let regex: NSRegularExpression
    
    public init(pattern: String, mode: Mode = .full, exclude: Bool = false) throws {
        self.pattern = pattern
        self.mode = mode
        self.exclude = exclude
        self.regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }
    
    public func validate(_ text: String) -> Bool {
        let range = NSRange(text.startIndex..., in: text)
        let matched: Bool
        
        switch mode {
        case .full:
            if let match = regex.firstMatch(in: text, range: range) {
                matched = match.range == range
            } else {
                matched = false
            }
        case .partial:
            matched = regex.firstMatch(in: text, range: range) != nil
        }
        
        return exclude ? !matched : matched
    }
}
