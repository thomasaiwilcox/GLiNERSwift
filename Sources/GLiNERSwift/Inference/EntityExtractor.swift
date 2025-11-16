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
        // Early return for empty input
        guard !spanScores.isEmpty else {
            return []
        }
        
        // Filter by threshold - use filter with stride for better performance
        let filtered = spanScores.filter { $0.score >= config.threshold }
        
        // Early return if nothing passes threshold
        guard !filtered.isEmpty else {
            return []
        }
        
        // Sort by score (descending) - already optimized by Swift
        let sorted = filtered.sorted { $0.score > $1.score }
        
        // Apply non-maximum suppression with optimized algorithm
        let afterNMS = applyNMS(sorted)
        
        // Convert to Entity objects with pre-allocated capacity
        var entities: [Entity] = []
        entities.reserveCapacity(afterNMS.count)
        
        for spanScore in afterNMS {
            if let entity = convertToEntity(spanScore, text: text, tokens: tokens) {
                entities.append(entity)
            }
        }
        
        return entities
    }
    
    /// Apply optimized non-maximum suppression to remove overlapping spans
    private func applyNMS(_ spanScores: [SpanScore]) -> [SpanScore] {
        guard !spanScores.isEmpty else {
            return []
        }
        
        var selected: [SpanScore] = []
        selected.reserveCapacity(spanScores.count / 2) // Heuristic pre-allocation
        
        // Group by label for more efficient NMS
        var byLabel: [String: [SpanScore]] = [:]
        for score in spanScores {
            byLabel[score.label, default: []].append(score)
        }
        
        // Process each label independently (allows for parallel processing in future)
        for (_, labelScores) in byLabel {
            var remaining = labelScores
            var labelSelected: [SpanScore] = []
            
            while !remaining.isEmpty {
                // Take the highest scoring span
                let best = remaining.removeFirst()
                labelSelected.append(best)
                
                // Fast path: pre-compute best span bounds
                let bestStart = best.span.start
                let bestEnd = best.span.end
                
                // Remove overlapping spans using optimized IoU computation
                remaining.removeAll { current in
                    let iou = computeIoUFast(
                        start1: bestStart, end1: bestEnd,
                        start2: current.span.start, end2: current.span.end
                    )
                    return iou > config.nmsThreshold
                }
            }
            
            selected.append(contentsOf: labelSelected)
        }
        
        // Re-sort by score to maintain best-first order
        return selected.sorted { $0.score > $1.score }
    }
    
    /// Optimized IoU computation with inlined bounds checking
    @inline(__always)
    private func computeIoUFast(
        start1: Int, end1: Int,
        start2: Int, end2: Int
    ) -> Float {
        let intersectionStart = max(start1, start2)
        let intersectionEnd = min(end1, end2)
        
        guard intersectionStart < intersectionEnd else {
            return 0.0
        }
        
        let intersection = intersectionEnd - intersectionStart
        let union = (end1 - start1) + (end2 - start2) - intersection
        
        // Avoid division by zero
        guard union > 0 else {
            return 0.0
        }
        
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
