import Foundation

/// Builds span score tensors using GLiNER2's structure-aware projections.
struct GLiNER2SpanScoreBuilder {
    func buildScores(
        spanEmbeddings: [[[Float]]],
        structureEmbeddings: [[[Float]]],
        spanMask: [[Float]],
        predictedCount: Int
    ) -> SpanScores {
        guard predictedCount > 0,
              !spanEmbeddings.isEmpty,
              let maxWidth = spanEmbeddings.first?.count,
              maxWidth > 0,
              let firstInstance = structureEmbeddings.first,
              !firstInstance.isEmpty else {
            return SpanScores(values: [])
        }

        let labelEmbeddings = firstInstance
        let hiddenSize = labelEmbeddings.first?.count ?? 0
        guard hiddenSize > 0 else {
            return SpanScores(values: [])
        }

        if spanEmbeddings.count < 40,
           let firstSpan = spanEmbeddings.first?.first,
           let firstLabel = labelEmbeddings.first {
            let spanNorm = sqrt(firstSpan.reduce(0) { $0 + $1 * $1 })
            let labelNorm = sqrt(firstLabel.reduce(0) { $0 + $1 * $1 })
            let spanMax = firstSpan.map { abs($0) }.max() ?? 0
            let labelMax = firstLabel.map { abs($0) }.max() ?? 0
            print("[DEBUG] Span/label norms: span=\(spanNorm) label=\(labelNorm) spanMax=\(spanMax) labelMax=\(labelMax)")
        }

        var scores = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: labelEmbeddings.count),
                count: maxWidth
            ),
            count: spanEmbeddings.count
        )

        for (wordIndex, widths) in spanEmbeddings.enumerated() {
            let maskRow = wordIndex < spanMask.count ? spanMask[wordIndex] : []
            for (widthIndex, spanVector) in widths.enumerated() {
                guard widthIndex < maskRow.count, maskRow[widthIndex] > 0 else {
                    continue
                }
                guard spanVector.count == hiddenSize else { continue }
                for (labelIndex, labelVector) in labelEmbeddings.enumerated() where labelVector.count == hiddenSize {
                    scores[wordIndex][widthIndex][labelIndex] = dotProduct(spanVector, labelVector)
                }
            }
        }

        return SpanScores(values: scores)
    }
}

private extension GLiNER2SpanScoreBuilder {
    func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        var result: Float = 0
        let count = min(lhs.count, rhs.count)
        for idx in 0..<count {
            result += lhs[idx] * rhs[idx]
        }
        return result
    }
}
