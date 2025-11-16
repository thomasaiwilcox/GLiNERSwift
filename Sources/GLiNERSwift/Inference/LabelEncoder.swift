import Foundation
import Accelerate

/// Encodes labels into embeddings and caches them
public class LabelEncoder {
    private let encoder: GLiNEREncoder
    private let tokenizer: GLiNERTokenizer
    private let config: Configuration
    
    // Thread-safe cache for label embeddings
    private var cache: [String: [Float]] = [:]
    private let cacheQueue = DispatchQueue(label: "com.glinerswift.labelencoder.cache")
    
    public init(
        encoder: GLiNEREncoder,
        tokenizer: GLiNERTokenizer,
        config: Configuration = .default
    ) {
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
    }
    
    /// Encode a single label to an embedding vector
    /// - Parameter label: Label text (e.g., "person", "organization")
    /// - Returns: Label embedding vector
    public func encodeLabel(_ label: String) async throws -> [Float] {
        // Check cache first
        if let cached = getCached(label: label) {
            return cached
        }
        
        // Encode label text
        let tokenized = try tokenizer.encode(label)
        let hiddenStates = try await encoder.encode(
            inputIds: tokenized.inputIds,
            attentionMask: tokenized.attentionMask
        )
        
        // Pool to get label embedding (mean pooling over valid tokens)
        let validTokenCount = tokenized.attentionMask.prefix(while: { $0 == 1 }).count
        let labelEmbedding = meanPool(
            hiddenStates: hiddenStates,
            validTokenCount: validTokenCount
        )
        
        // Cache the result
        setCached(label: label, embedding: labelEmbedding)
        
        return labelEmbedding
    }
    
    /// Encode multiple labels at once
    /// - Parameter labels: Array of label texts
    /// - Returns: Dictionary mapping labels to embeddings
    public func encodeLabels(_ labels: [String]) async throws -> [String: [Float]] {
        var result: [String: [Float]] = [:]
        
        for label in labels {
            result[label] = try await encodeLabel(label)
        }
        
        return result
    }
    
    /// Clear the label cache
    public func clearCache() {
        cacheQueue.sync {
            cache.removeAll()
        }
    }
    
    // MARK: - Private Methods
    
    private func getCached(label: String) -> [Float]? {
        cacheQueue.sync {
            cache[label]
        }
    }
    
    private func setCached(label: String, embedding: [Float]) {
        cacheQueue.sync {
            cache[label] = embedding
        }
    }
    
    private func meanPool(hiddenStates: [[Float]], validTokenCount: Int) -> [Float] {
        guard validTokenCount > 0, validTokenCount <= hiddenStates.count else {
            return []
        }
        
        let hiddenDim = hiddenStates[0].count
        var result = [Float](repeating: 0, count: hiddenDim)
        
        // Sum valid token embeddings
        for tokenIdx in 0..<validTokenCount {
            let embedding = hiddenStates[tokenIdx]
            vDSP_vadd(result, 1, embedding, 1, &result, 1, vDSP_Length(hiddenDim))
        }
        
        // Divide by token count
        var divisor = Float(validTokenCount)
        vDSP_vsdiv(result, 1, &divisor, &result, 1, vDSP_Length(hiddenDim))
        
        return result
    }
}
