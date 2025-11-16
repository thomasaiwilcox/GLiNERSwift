import Foundation

/// Extracts entities from scored spans
public class EntityExtractor {
    private let config: Configuration
    
    public init(config: Configuration = .default) {
        self.config = config
    }
    
    /// Extract entities from span scores
    /// - Parameters:
    ///   - spanScores: Scored span-label pairs
    ///   - text: Original text
    ///   - tokens: Tokenized text
    /// - Returns: Array of extracted entities
    public func extractEntities(
        from spanScores: [SpanScore],
        text: String,
        tokens: [String]
    ) -> [Entity] {
        // Filter by threshold
        let filtered = spanScores.filter { score in
            return score.score >= config.threshold
        }
        
        // Sort by score (descending)
        let sorted = filtered.sorted { $0.score > $1.score }
        
        // Apply non-maximum suppression
        let afterNMS = applyNMS(sorted)
        
        // Convert to Entity objects
        return afterNMS.compactMap { spanScore in
            convertToEntity(spanScore, text: text, tokens: tokens)
        }
    }
    
    /// Apply non-maximum suppression to remove overlapping spans
    private func applyNMS(_ spanScores: [SpanScore]) -> [SpanScore] {
        var selected: [SpanScore] = []
        var remaining = spanScores
        
        while !remaining.isEmpty {
            // Take the highest scoring span
            let best = remaining.removeFirst()
            selected.append(best)
            
            // Remove overlapping spans for the same label
            remaining.removeAll { current in
                guard current.label == best.label else {
                    return false
                }
                
                let iou = computeIoU(
                    span1: (best.span.start, best.span.end),
                    span2: (current.span.start, current.span.end)
                )
                
                return iou > config.nmsThreshold
            }
        }
        
        return selected
    }
    
    /// Compute Intersection over Union for two spans
    private func computeIoU(
        span1: (start: Int, end: Int),
        span2: (start: Int, end: Int)
    ) -> Float {
        let intersectionStart = max(span1.start, span2.start)
        let intersectionEnd = min(span1.end, span2.end)
        
        guard intersectionStart < intersectionEnd else {
            return 0.0
        }
        
        let intersection = intersectionEnd - intersectionStart
        let union = (span1.end - span1.start) + (span2.end - span2.start) - intersection
        
        return Float(intersection) / Float(union)
    }
    
    /// Convert SpanScore to Entity
    private func convertToEntity(
        _ spanScore: SpanScore,
        text: String,
        tokens: [String]
    ) -> Entity? {
        // Extract token span
        let spanTokens = Array(tokens[spanScore.span.start..<spanScore.span.end])
        
        // Reconstruct text from tokens (simplified - proper implementation needs detokenization)
        let entityText = spanTokens
            .joined(separator: " ")
            .replacingOccurrences(of: " ##", with: "") // Handle WordPiece
            .replacingOccurrences(of: "##", with: "")
        
        // Try to find the entity text in the original text
        // This is a simplified approach - proper implementation would use token-to-char mapping
        guard let range = text.range(of: entityText, options: .caseInsensitive) else {
            return nil
        }
        
        let start = text.distance(from: text.startIndex, to: range.lowerBound)
        let end = text.distance(from: text.startIndex, to: range.upperBound)
        
        return Entity(
            text: String(text[range]),
            label: spanScore.label,
            score: spanScore.score,
            start: start,
            end: end
        )
    }
}
