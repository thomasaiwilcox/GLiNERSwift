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
        guard validTokenCount > 0 else {
            return []
        }
        
        var spans: [Span] = []
        
        // Enumerate all possible spans up to maxSpanLength
        for start in 0..<validTokenCount {
            for length in 1...min(config.maxSpanLength, validTokenCount - start) {
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
        var result = [Float](repeating: 0, count: hiddenDim)
        
        // Sum all token embeddings in the span
        for tokenIdx in start..<end {
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
        
        let hiddenDim = hiddenStates[0].count
        var result = hiddenStates[start]
        
        // Take max across all dimensions
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
