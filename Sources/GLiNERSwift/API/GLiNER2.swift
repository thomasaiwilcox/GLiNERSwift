import Foundation
@preconcurrency import CoreML

/// GLiNER2 high-level API with Schema builder support, matching Python's GLiNER2 class.
public class GLiNER2 {
    private let model: GLiNERModel
    private let defaultConfig: Configuration
    
    /// Initialize GLiNER2 from a manifest (typical usage).
    /// - Parameters:
    ///   - manifestURL: Path to export_manifest.json
    ///   - config: Optional runtime configuration
    ///   - encoderComputeUnits: Compute units for encoder
    public init(
        manifestURL: URL,
        config: Configuration = .default,
        encoderComputeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        self.model = try await GLiNERModel(
            manifestURL: manifestURL,
            config: config,
            encoderComputeUnits: encoderComputeUnits
        )
        self.defaultConfig = config
    }
    
    /// Initialize GLiNER2 from custom model/tokenizer URLs.
    /// - Parameters:
    ///   - modelURL: Path to encoder .mlpackage
    ///   - tokenizerURL: Path to tokenizer directory
    ///   - config: Optional runtime configuration
    ///   - encoderComputeUnits: Compute units for encoder
    public init(
        modelURL: URL,
        tokenizerURL: URL,
        config: Configuration = .default,
        encoderComputeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        self.model = try await GLiNERModel(
            modelURL: modelURL,
            tokenizerURL: tokenizerURL,
            config: config,
            encoderComputeUnits: encoderComputeUnits
        )
        self.defaultConfig = config
    }
    
    /// Convenience: Extract entities from text with simple label list.
    /// Equivalent to Python's `gliner.extract_entities(text, labels)`.
    /// - Parameters:
    ///   - text: Input text
    ///   - labels: Entity types to extract
    ///   - threshold: Optional confidence threshold
    /// - Returns: Extracted entities
    public func extractEntities(
        from text: String,
        labels: [String],
        threshold: Float? = nil
    ) async throws -> [Entity] {
        return try await model.extractEntities(
            from: text,
            labels: labels,
            threshold: threshold
        )
    }
    
    /// Multi-task extraction using Schema builder.
    /// Supports entities, classification, and structured extraction.
    /// Equivalent to Python's `gliner.extract(text, schema)`.
    /// - Parameters:
    ///   - text: Input text
    ///   - schema: Schema defining extraction tasks
    ///   - threshold: Optional global confidence threshold
    /// - Returns: ExtractionResult with all extracted data
    public func extract(
        from text: String,
        schema: Schema,
        threshold: Float? = nil
    ) async throws -> ExtractionResult {
        let schemaDict = schema.build()
        
        var result = ExtractionResult()
        
        // Entity extraction
        if let entitiesList = schemaDict["entities"] as? [[String: Any]] {
            let labels = entitiesList.compactMap { $0["entity"] as? String }
            if !labels.isEmpty {
                let entities = try await model.extractEntities(
                    from: text,
                    labels: labels,
                    threshold: threshold
                )
                
                // Apply per-entity thresholds and dtype conversions
                var groupedEntities: [String: [Entity]] = [:]
                for entity in entities {
                    if let metadata = schema.entityMetadata[entity.label] {
                        // Apply entity-specific threshold if set
                        if let entityThreshold = metadata.threshold, entity.score < entityThreshold {
                            continue
                        }
                    }
                    groupedEntities[entity.label, default: []].append(entity)
                }
                
                // Convert based on dtype
                for label in labels {
                    let metadata = schema.entityMetadata[label]
                    let entityList = groupedEntities[label] ?? []
                    
                    if metadata?.dtype == "str" {
                        // Return first entity text only
                        result.entities[label] = entityList.first?.text ?? ""
                    } else {
                        // Return list of entity texts
                        result.entities[label] = entityList.map { $0.text }
                    }
                }
            }
        }
        
        // Text classification
        if let classificationsList = schemaDict["classifications"] as? [[String: Any]] {
            for classificationTask in classificationsList {
                guard let taskName = classificationTask["task"] as? String,
                      let labels = classificationTask["labels"] as? [String] else {
                    continue
                }
                
                let multiLabel = classificationTask["multi_label"] as? Bool ?? false
                let clsThreshold = classificationTask["cls_threshold"] as? Float ?? 0.5
                
                let predictions = try await model.classifyText(
                    text,
                    labels: labels,
                    multiLabel: multiLabel,
                    threshold: clsThreshold
                )
                
                let classificationPredictions = predictions.map {
                    ClassificationResult.Prediction(label: $0.label, score: $0.score)
                }
                
                result.classifications[taskName] = ClassificationResult(predictions: classificationPredictions)
            }
        }
        
        // Structured extraction (JSON structures)
        if let structuresList = schemaDict["json_structures"] as? [[String: [String: Any]]] {
            for structureDict in structuresList {
                for (structureName, fields) in structureDict {
                    let extracted = try await extractStructure(
                        from: text,
                        structureName: structureName,
                        fields: fields,
                        schema: schema
                    )
                    result.structures[structureName] = extracted
                }
            }
        }
        
        return result
    }
    
    /// Batch extraction with dynamic padding.
    /// Equivalent to Python's `gliner.batch_extract(texts, schema)`.
    /// - Parameters:
    ///   - texts: Array of input texts
    ///   - schema: Schema defining extraction tasks
    ///   - threshold: Optional global confidence threshold
    /// - Returns: Array of ExtractionResults
    public func batchExtract(
        from texts: [String],
        schema: Schema,
        threshold: Float? = nil
    ) async throws -> [ExtractionResult] {
        // TODO: Implement true batch processing with dynamic padding
        // For now, process sequentially
        var results: [ExtractionResult] = []
        for text in texts {
            let result = try await extract(from: text, schema: schema, threshold: threshold)
            results.append(result)
        }
        return results
    }
    
    /// Classify text into one or more labels.
    /// Equivalent to Python's `gliner.classify_text(text, labels)`.
    /// - Parameters:
    ///   - text: Input text
    ///   - task: Classification task name
    ///   - labels: Classification labels
    ///   - multiLabel: Allow multiple labels
    ///   - threshold: Confidence threshold
    /// - Returns: Classification result
    public func classifyText(
        _ text: String,
        task: String,
        labels: [String],
        multiLabel: Bool = false,
        threshold: Float = 0.5
    ) async throws -> ClassificationResult {
        let predictions = try await model.classifyText(
            text,
            labels: labels,
            multiLabel: multiLabel,
            threshold: threshold
        )
        
        let classificationPredictions = predictions.map {
            ClassificationResult.Prediction(label: $0.label, score: $0.score)
        }
        
        return ClassificationResult(predictions: classificationPredictions)
    }
    
    /// Extract structured JSON data from text.
    /// Equivalent to Python's `gliner.extract_json(text, schema)`.
    /// - Parameters:
    ///   - text: Input text
    ///   - schema: Schema with structure definitions
    /// - Returns: Structured data as dictionary
    public func extractJSON(
        from text: String,
        schema: Schema
    ) async throws -> [String: Any] {
        let result = try await extract(from: text, schema: schema)
        return result.structures
    }
    
    // MARK: - Private helpers
    
    private func extractStructure(
        from text: String,
        structureName: String,
        fields: [String: Any],
        schema: Schema
    ) async throws -> [String: Any] {
        var extracted: [String: Any] = [:]
        let fieldOrder = schema.fieldOrders[structureName] ?? Array(fields.keys)
        
        for fieldName in fieldOrder {
            guard let fieldConfig = fields[fieldName] as? [String: Any] else {
                continue
            }
            
            let fieldKey = "\(structureName).\(fieldName)"
            let metadata = schema.fieldMetadata[fieldKey]
            
            // Check if field has choices (selection mode)
            if let choices = fieldConfig["choices"] as? [String] {
                // Extract using [selection] prefix
                let entities = try await model.extractEntities(
                    from: text,
                    labels: choices,
                    threshold: metadata?.threshold
                )
                
                // Apply validators if present
                var validEntities = entities
                if let validators = metadata?.validators {
                    validEntities = entities.filter { entity in
                        validators.allSatisfy { $0.validate(entity.text) }
                    }
                }
                
                if let dtype = metadata?.dtype, dtype == "str" {
                    extracted[fieldName] = validEntities.first?.text ?? ""
                } else {
                    extracted[fieldName] = validEntities.map { $0.text }
                }
            } else {
                // Regular entity extraction for the field
                let description = fieldConfig["description"] as? String ?? fieldName
                let entities = try await model.extractEntities(
                    from: text,
                    labels: [description],
                    threshold: metadata?.threshold
                )
                
                // Apply validators if present
                var validEntities = entities
                if let validators = metadata?.validators {
                    validEntities = entities.filter { entity in
                        validators.allSatisfy { $0.validate(entity.text) }
                    }
                }
                
                if let dtype = metadata?.dtype, dtype == "str" {
                    extracted[fieldName] = validEntities.first?.text ?? ""
                } else {
                    extracted[fieldName] = validEntities.map { $0.text }
                }
            }
        }
        
        return extracted
    }
}

/// Result from multi-task extraction
public struct ExtractionResult {
    public var entities: [String: Any] = [:]
    public var classifications: [String: ClassificationResult] = [:]
    public var structures: [String: Any] = [:]
    
    public init() {}
}

/// Classification result with predictions and scores
public struct ClassificationResult {
    public struct Prediction {
        public let label: String
        public let score: Float
        
        public init(label: String, score: Float) {
            self.label = label
            self.score = score
        }
    }
    
    public let predictions: [Prediction]
    
    public init(predictions: [Prediction]) {
        self.predictions = predictions
    }
    
    /// Get the top prediction
    public var topPrediction: Prediction? {
        predictions.max(by: { $0.score < $1.score })
    }
    
    /// Get all labels above threshold
    public func labels(aboveThreshold threshold: Float) -> [String] {
        predictions.filter { $0.score >= threshold }.map { $0.label }
    }
}
