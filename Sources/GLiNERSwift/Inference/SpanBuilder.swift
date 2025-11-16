import Foundation
import Accelerate

/// Represents a candidate span in the text
public struct Span {
    let start: Int  // Token index
    let end: Int    // Token index (exclusive)
    let embedding: [Float]
    
    var length: Int {
        end - start
    }
}

/// Builds span representations from token embeddings
public class SpanBuilder {
    private let config: Configuration
    // Reusable buffer for pooling operations to reduce allocations
    private var poolingBuffer: [Float] = []
    
    public init(config: Configuration = .default) {
        self.config = config
    }
    
    /// Build all candidate spans from hidden states
    /// - Parameters:
    ///   - hiddenStates: Token embeddings [sequenceLength, hiddenDim]
    ///   - attentionMask: Attention mask to identify valid tokens
    /// - Returns: Array of Span objects
    public func buildSpans(
        from hiddenStates: [[Float]],
        attentionMask: [Int32]
    ) -> [Span] {
        let sequenceLength = hiddenStates.count
        guard sequenceLength == attentionMask.count else {
            return []
        }
        
        // Find valid token range (exclude padding)
        let validTokenCount = attentionMask.prefix(while: { $0 == 1 }).count
        guard validTokenCount > 0, !hiddenStates.isEmpty else {
            return []
        }
        
        // Pre-allocate spans array with estimated capacity to reduce reallocations
        let estimatedSpanCount = min(validTokenCount * config.maxSpanLength / 2, 10000)
        var spans: [Span] = []
        spans.reserveCapacity(estimatedSpanCount)
        
        // Ensure pooling buffer is sized appropriately
        let hiddenDim = hiddenStates[0].count
        if poolingBuffer.count != hiddenDim {
            poolingBuffer = [Float](repeating: 0, count: hiddenDim)
        }
        
        // Enumerate all possible spans up to maxSpanLength
        for start in 0..<validTokenCount {
            let maxLength = min(config.maxSpanLength, validTokenCount - start)
            for length in 1...maxLength {
                let end = start + length
                
                // Skip if this would exceed valid tokens
                guard end <= validTokenCount else { break }
                
                // Build span embedding based on pooling method
                let embedding = buildSpanEmbedding(
                    hiddenStates: hiddenStates,
                    start: start,
                    end: end
                )
                
                spans.append(Span(start: start, end: end, embedding: embedding))
            }
        }
        
        return spans
    }
    
    // MARK: - Private Methods
    
    private func buildSpanEmbedding(
        hiddenStates: [[Float]],
        start: Int,
        end: Int
    ) -> [Float] {
        switch config.poolingMethod {
        case .mean:
            return meanPool(hiddenStates: hiddenStates, start: start, end: end)
        case .max:
            return maxPool(hiddenStates: hiddenStates, start: start, end: end)
        case .concat:
            return concatPool(hiddenStates: hiddenStates, start: start, end: end)
        }
    }
    
    private func meanPool(
        hiddenStates: [[Float]],
        start: Int,
        end: Int
    ) -> [Float] {
        guard start < end, end <= hiddenStates.count else {
            return []
        }
        
        let hiddenDim = hiddenStates[0].count
        let spanLength = end - start
        
        // Optimize for single-token spans (common case)
        if spanLength == 1 {
            return hiddenStates[start]
        }
        
        // Use pooling buffer to reduce allocations
        var result = poolingBuffer
        
        // Initialize with first embedding
        result.withUnsafeMutableBufferPointer { resultPtr in
            hiddenStates[start].withUnsafeBufferPointer { firstPtr in
                resultPtr.baseAddress?.initialize(from: firstPtr.baseAddress!, count: hiddenDim)
            }
        }
        
        // Sum remaining token embeddings in the span using vectorized addition
        for tokenIdx in (start + 1)..<end {
            let embedding = hiddenStates[tokenIdx]
            vDSP_vadd(result, 1, embedding, 1, &result, 1, vDSP_Length(hiddenDim))
        }
        
        // Divide by span length to get mean
        var divisor = Float(spanLength)
        vDSP_vsdiv(result, 1, &divisor, &result, 1, vDSP_Length(hiddenDim))
        
        return result
    }
    
    private func maxPool(
        hiddenStates: [[Float]],
        start: Int,
        end: Int
    ) -> [Float] {
        guard start < end, end <= hiddenStates.count else {
            return []
        }
        
        let spanLength = end - start
        
        // Optimize for single-token spans
        if spanLength == 1 {
            return hiddenStates[start]
        }
        
        let hiddenDim = hiddenStates[0].count
        var result = hiddenStates[start]
        
        // Take max across all dimensions using vectorized operations
        for tokenIdx in (start + 1)..<end {
            let embedding = hiddenStates[tokenIdx]
            vDSP_vmax(result, 1, embedding, 1, &result, 1, vDSP_Length(hiddenDim))
        }
        
        return result
    }
    
    private func concatPool(
        hiddenStates: [[Float]],
        start: Int,
        end: Int
    ) -> [Float] {
        guard start < end, end <= hiddenStates.count else {
            return []
        }
        
        // Concatenate: [start_embedding, end_embedding, mean_pooled]
        let startEmbedding = hiddenStates[start]
        let endEmbedding = hiddenStates[end - 1]
        let meanEmbedding = meanPool(hiddenStates: hiddenStates, start: start, end: end)
        
        return startEmbedding + endEmbedding + meanEmbedding
    }
}
