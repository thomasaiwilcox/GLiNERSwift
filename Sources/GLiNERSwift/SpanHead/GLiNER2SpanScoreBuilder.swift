import Foundation
import Accelerate

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

        // Pre-allocate scores array
        var scores = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: labelEmbeddings.count),
                count: maxWidth
            ),
            count: spanEmbeddings.count
        )

        // Pre-flatten label embeddings for better cache locality
        let numLabels = labelEmbeddings.count
        let flatLabelEmbeddings = labelEmbeddings.flatMap { $0 }
        
        // Process spans with vectorized operations
        for (wordIndex, widths) in spanEmbeddings.enumerated() {
            let maskRow = wordIndex < spanMask.count ? spanMask[wordIndex] : []
            for (widthIndex, spanVector) in widths.enumerated() {
                guard widthIndex < maskRow.count, maskRow[widthIndex] > 0 else {
                    continue
                }
                guard spanVector.count == hiddenSize else { continue }
                
                // Vectorized batch computation of all label similarities for this span
                scores[wordIndex][widthIndex] = batchDotProduct(
                    spanVector: spanVector,
                    labelEmbeddings: flatLabelEmbeddings,
                    numLabels: numLabels,
                    hiddenSize: hiddenSize
                )
            }
        }

        return SpanScores(values: scores)
    }
}

private extension GLiNER2SpanScoreBuilder {
    /// Compute dot product using Accelerate framework for better performance
    func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0.0 }
        
        var result: Float = 0
        vDSP_dotpr(lhs, 1, rhs, 1, &result, vDSP_Length(count))
        return result
    }
    
    /// Batch compute dot products between one span and all labels using matrix-vector multiplication
    func batchDotProduct(
        spanVector: [Float],
        labelEmbeddings: [Float],
        numLabels: Int,
        hiddenSize: Int
    ) -> [Float] {
        guard hiddenSize > 0, numLabels > 0 else { return [] }
        
        var scores = [Float](repeating: 0, count: numLabels)
        
        // Use BLAS matrix-vector multiplication: scores = labelEmbeddings * spanVector
        // labelEmbeddings is treated as (numLabels x hiddenSize) matrix in row-major order
        // spanVector is (hiddenSize x 1) vector
        // Result is (numLabels x 1) vector
        cblas_sgemv(
            CblasRowMajor,          // Row-major order
            CblasNoTrans,           // Don't transpose matrix
            Int32(numLabels),       // Number of rows
            Int32(hiddenSize),      // Number of columns
            1.0,                    // Alpha (scaling factor)
            labelEmbeddings,        // Matrix A
            Int32(hiddenSize),      // Leading dimension of A
            spanVector,             // Vector x
            1,                      // Increment for x
            0.0,                    // Beta (scaling factor for y)
            &scores,                // Result vector y
            1                       // Increment for y
        )
        
        return scores
    }
}
