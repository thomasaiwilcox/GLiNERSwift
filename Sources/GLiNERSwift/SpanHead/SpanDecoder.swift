import Foundation

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

        var candidates: [SpanCandidate] = []
        for (startWord, widths) in scores.values.enumerated() {
            guard startWord < wordRanges.count else { break }
            for (widthIndex, classScores) in widths.enumerated() {
                let endWord = startWord + widthIndex
                if endWord >= wordRanges.count {
                    break
                }
                for labelIndex in 0..<min(classCount, classScores.count) {
                    let probability = sigmoid(classScores[labelIndex])
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
        return selected.compactMap { candidate in
            guard candidate.startWord >= 0,
                  candidate.endWord < wordRanges.count,
                  candidate.labelIndex < labels.count else { return nil }
            let startRange = wordRanges[candidate.startWord]
            let endRange = wordRanges[candidate.endWord]
            let lowerBound = startRange.lowerBound
            let upperBound = endRange.upperBound
            let entityStart = text.distance(from: text.startIndex, to: lowerBound)
            let entityEnd = text.distance(from: text.startIndex, to: upperBound)
            let surfaceForm = String(text[lowerBound..<upperBound])
            return Entity(
                text: surfaceForm,
                label: labels[candidate.labelIndex],
                score: candidate.score,
                start: entityStart,
                end: entityEnd
            )
        }
    }

    private func greedySelect(_ candidates: [SpanCandidate]) -> [SpanCandidate] {
        var chosen: [SpanCandidate] = []
        let sorted = candidates.sorted { $0.score > $1.score }
        outer: for candidate in sorted {
            for existing in chosen where overlaps(candidate, existing) {
                continue outer
            }
            chosen.append(candidate)
        }
        return chosen.sorted { lhs, rhs in
            if lhs.startWord == rhs.startWord {
                return lhs.score > rhs.score
            }
            return lhs.startWord < rhs.startWord
        }
    }

    private func overlaps(_ lhs: SpanCandidate, _ rhs: SpanCandidate) -> Bool {
        if lhs.startWord == rhs.startWord && lhs.endWord == rhs.endWord {
            return true
        }
        if lhs.startWord > rhs.endWord || rhs.startWord > lhs.endWord {
            return false
        }
        return true
    }

    private func sigmoid(_ value: Float) -> Float {
        1.0 / (1.0 + exp(-value))
    }
}
