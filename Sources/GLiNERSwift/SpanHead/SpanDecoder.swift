import Foundation
import Accelerate

private struct SpanCandidate {
    let startWord: Int
    let endWord: Int // Inclusive index
    let labelIndex: Int
    let score: Float
}

/// Applies GLiNER's greedy span decoding with sigmoid thresholding and overlap suppression.
final class SpanDecoder {
    func decode(
        scores: SpanScores,
        labels: [String],
        threshold: Float,
        text: String,
        wordRanges: [Range<String.Index>]
    ) -> [Entity] {
        guard !scores.values.isEmpty else { return [] }
        let widthCount = scores.values.first?.count ?? 0
        guard widthCount > 0 else { return [] }
        guard !wordRanges.isEmpty else { return [] }
        let classCount = labels.count
        guard classCount > 0 else { return [] }

        // Pre-allocate candidates array with estimated capacity
        let estimatedCandidates = scores.values.count * widthCount * classCount / 10
        var candidates: [SpanCandidate] = []
        candidates.reserveCapacity(estimatedCandidates)
        
        // Extract candidates with early threshold filtering
        for (startWord, widths) in scores.values.enumerated() {
            guard startWord < wordRanges.count else { break }
            
            for (widthIndex, classScores) in widths.enumerated() {
                let endWord = startWord + widthIndex
                if endWord >= wordRanges.count {
                    break
                }
                
                // Process all class scores for this span
                let labelCount = min(classCount, classScores.count)
                for labelIndex in 0..<labelCount {
                    let rawScore = classScores[labelIndex]
                    let probability = sigmoid(rawScore)
                    
                    if probability >= threshold {
                        candidates.append(SpanCandidate(
                            startWord: startWord,
                            endWord: endWord,
                            labelIndex: labelIndex,
                            score: probability
                        ))
                    }
                }
            }
        }

        guard !candidates.isEmpty else { return [] }
        
        let selected = greedySelect(candidates)
        
        // Pre-allocate entities array
        var entities: [Entity] = []
        entities.reserveCapacity(selected.count)
        
        for candidate in selected {
            guard candidate.startWord >= 0,
                  candidate.endWord < wordRanges.count,
                  candidate.labelIndex < labels.count else { continue }
            
            let startRange = wordRanges[candidate.startWord]
            let endRange = wordRanges[candidate.endWord]
            let lowerBound = startRange.lowerBound
            let upperBound = endRange.upperBound
            let entityStart = text.distance(from: text.startIndex, to: lowerBound)
            let entityEnd = text.distance(from: text.startIndex, to: upperBound)
            let surfaceForm = String(text[lowerBound..<upperBound])
            
            entities.append(Entity(
                text: surfaceForm,
                label: labels[candidate.labelIndex],
                score: candidate.score,
                start: entityStart,
                end: entityEnd
            ))
        }
        
        return entities
    }

    private func greedySelect(_ candidates: [SpanCandidate]) -> [SpanCandidate] {
        guard !candidates.isEmpty else { return [] }
        
        var chosen: [SpanCandidate] = []
        chosen.reserveCapacity(candidates.count / 2) // Heuristic pre-allocation
        
        // Sort by score descending (already optimized by Swift)
        let sorted = candidates.sorted { $0.score > $1.score }
        
        // Greedy selection with optimized overlap check
        outer: for candidate in sorted {
            // Check overlap with all chosen candidates
            for existing in chosen {
                if overlapsFast(candidate, existing) {
                    continue outer
                }
            }
            chosen.append(candidate)
        }
        
        // Sort by start position, then by score
        return chosen.sorted { lhs, rhs in
            if lhs.startWord == rhs.startWord {
                return lhs.score > rhs.score
            }
            return lhs.startWord < rhs.startWord
        }
    }

    /// Optimized overlap check with inlining
    @inline(__always)
    private func overlapsFast(_ lhs: SpanCandidate, _ rhs: SpanCandidate) -> Bool {
        // Fast exact match check
        if lhs.startWord == rhs.startWord && lhs.endWord == rhs.endWord {
            return true
        }
        
        // Fast non-overlap check (most common case)
        if lhs.startWord > rhs.endWord || rhs.startWord > lhs.endWord {
            return false
        }
        
        // Any other case is an overlap
        return true
    }

    /// Vectorized sigmoid computation using Accelerate
    private func sigmoid(_ value: Float) -> Float {
        // For single values, simple computation is faster than vDSP overhead
        return 1.0 / (1.0 + exp(-value))
    }
}
