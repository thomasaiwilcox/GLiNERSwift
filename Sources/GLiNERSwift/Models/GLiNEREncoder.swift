import Foundation
import Accelerate
@preconcurrency import CoreML

/// Wrapper around the Core ML GLiNER encoder model.
public class GLiNEREncoder {
    private let model: MLModel
    private let configuration: MLModelConfiguration
    private let outputFeatureName: String
    private let predictionQueue = DispatchQueue(label: "com.glinerswift.encoder.prediction", qos: .userInitiated)
    
    /// Initialize encoder with bundled model
    /// - Parameter computeUnits: Desired Core ML compute units (default: .cpuAndGPU)
    public init(computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        // Dynamic sequence lengths are unsupported on ANE, and the BNNS-only path currently crashes on macOS 15,
        // so run the encoder on CPU/GPU by default unless overridden.
        config.computeUnits = computeUnits
        
        let modelURL = try GLiNEREncoder.locateBundledModel(for: computeUnits)
        let compiledURL = try GLiNEREncoder.compileIfNeeded(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.configuration = config
        self.outputFeatureName = GLiNEREncoder.resolveOutputFeatureName(for: self.model)
    }
    
    /// Initialize encoder with custom model URL
    /// - Parameters:
    ///   - modelURL: URL to .mlpackage or .mlmodelc
    ///   - computeUnits: Compute units to use (default: .cpuAndGPU to keep dynamic shapes off ANE)
    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        let compiledURL = try GLiNEREncoder.compileIfNeeded(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.configuration = config
        self.outputFeatureName = GLiNEREncoder.resolveOutputFeatureName(for: self.model)
    }
    
    /// Encode input tokens to hidden states
    /// - Parameters:
    ///   - inputIds: Token IDs [sequenceLength]
    ///   - attentionMask: Attention mask [sequenceLength]
    /// - Returns: Hidden states [sequenceLength, hiddenDim]
    public func encode(inputIds: [Int32], attentionMask: [Int32]) async throws -> [[Float]] {
        guard inputIds.count == attentionMask.count else {
            throw GLiNERError.invalidInput("Input IDs and attention mask must have same length")
        }
        
        guard inputIds.count > 0 else {
            throw GLiNERError.invalidInput("Input cannot be empty")
        }
        
        // Run prediction on a background queue to avoid blocking callers
        let idsCopy = inputIds
        let maskCopy = attentionMask
        let sequenceLength = inputIds.count
        let output = try await withCheckedThrowingContinuation { continuation in
            predictionQueue.async {
                do {
                    let inputIdsArray = try self.createMLMultiArray(from: idsCopy, shape: [1, sequenceLength])
                    let attentionMaskArray = try self.createMLMultiArray(from: maskCopy, shape: [1, sequenceLength])
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                        "attention_mask": MLFeatureValue(multiArray: attentionMaskArray)
                    ])
                    let prediction = try self.model.prediction(from: provider)
                    continuation.resume(returning: prediction)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        
        // Extract hidden states
                guard let hiddenStatesValue = output.featureValue(for: outputFeatureName),
                            let hiddenStatesArray = hiddenStatesValue.multiArrayValue else {
                    let available = output.featureNames.joined(separator: ", ")
                    throw GLiNERError.invalidOutput("Could not extract hidden states from model output (expected \(outputFeatureName), available: [\(available)])")
        }
        
        // Convert to [[Float]] - shape is [1, sequenceLength, hiddenDim]
        return try extractHiddenStates(from: hiddenStatesArray)
    }
    
    // MARK: - Private Helpers
    
    private func createMLMultiArray(from array: [Int32], shape: [Int]) throws -> MLMultiArray {
        let mlArray = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        try mlArray.withUnsafeMutableFloat32Buffer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            for (index, value) in array.enumerated() {
                baseAddress[index] = Float(value)
            }
        }
        return mlArray
    }
    
    private func extractHiddenStates(from multiArray: MLMultiArray) throws -> [[Float]] {
        guard multiArray.shape.count == 3 else {
            throw GLiNERError.invalidOutput("Expected 3D tensor, got \(multiArray.shape.count)D")
        }
        let batchSize = multiArray.shape[0].intValue
        let sequenceLength = multiArray.shape[1].intValue
        let hiddenDim = multiArray.shape[2].intValue
        guard batchSize == 1 else {
            throw GLiNERError.invalidOutput("Expected batch size 1, got \(batchSize)")
        }
        switch multiArray.dataType {
        case .float32:
            return try multiArray.withUnsafeFloat32Buffer { buffer in
                try copyHiddenStates(
                    from: buffer,
                    strideSequence: multiArray.strides[1].intValue,
                    strideHidden: multiArray.strides[2].intValue,
                    sequenceLength: sequenceLength,
                    hiddenDim: hiddenDim
                )
            }
        case .float16:
            return try extractFloat16HiddenStates(from: multiArray, sequenceLength: sequenceLength, hiddenDim: hiddenDim)
        default:
            return try extractHiddenStatesFallback(from: multiArray, sequenceLength: sequenceLength, hiddenDim: hiddenDim)
        }
    }
    
    private func copyHiddenStates(
        from buffer: UnsafeBufferPointer<Float>,
        strideSequence: Int,
        strideHidden: Int,
        sequenceLength: Int,
        hiddenDim: Int
    ) throws -> [[Float]] {
        guard let baseAddress = buffer.baseAddress else { return [] }
        var result: [[Float]] = []
        result.reserveCapacity(sequenceLength)
        let contiguousHidden = strideHidden == 1
        for seqIdx in 0..<sequenceLength {
            var row = [Float](repeating: 0, count: hiddenDim)
            let sourceIndex = seqIdx * strideSequence
            if contiguousHidden {
                row.withUnsafeMutableBufferPointer { dest in
                    guard let destPtr = dest.baseAddress else { return }
                    destPtr.update(from: baseAddress.advanced(by: sourceIndex), count: hiddenDim)
                }
            } else {
                row.withUnsafeMutableBufferPointer { dest in
                    guard let destPtr = dest.baseAddress else { return }
                    var writePtr = destPtr
                    var readIndex = sourceIndex
                    for _ in 0..<hiddenDim {
                        writePtr.pointee = baseAddress[readIndex]
                        writePtr = writePtr.advanced(by: 1)
                        readIndex += strideHidden
                    }
                }
            }
            result.append(row)
        }
        return result
    }

    private func extractFloat16HiddenStates(from multiArray: MLMultiArray, sequenceLength: Int, hiddenDim: Int) throws -> [[Float]] {
        let strideSequence = multiArray.strides[1].intValue
        let strideHidden = multiArray.strides[2].intValue
        return try multiArray.withUnsafeFloat16Buffer { buffer in
            guard let baseAddress = buffer.baseAddress else { return [] }
            var result: [[Float]] = []
            result.reserveCapacity(sequenceLength)
            let contiguousHidden = strideHidden == 1
            var conversionScratch = [Float](repeating: 0, count: hiddenDim)
            for seqIdx in 0..<sequenceLength {
                let sourceIndex = seqIdx * strideSequence
                if contiguousHidden {
                    conversionScratch.withUnsafeMutableBufferPointer { dest in
                        guard let destPtr = dest.baseAddress else { return }
                        var floatBuffer = vImage_Buffer(data: destPtr, height: 1, width: vImagePixelCount(hiddenDim), rowBytes: hiddenDim * MemoryLayout<Float>.size)
                        var sourceBuffer = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: baseAddress.advanced(by: sourceIndex)),
                            height: 1,
                            width: vImagePixelCount(hiddenDim),
                            rowBytes: hiddenDim * MemoryLayout<UInt16>.size
                        )
                        let status = vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &floatBuffer, 0)
                        precondition(status == kvImageNoError, "vImage float16->float32 conversion failed with status \(status)")
                    }
                } else {
                    conversionScratch.withUnsafeMutableBufferPointer { dest in
                        guard let destPtr = dest.baseAddress else { return }
                        var writePtr = destPtr
                        var readIndex = sourceIndex
                        for _ in 0..<hiddenDim {
                            writePtr.pointee = Float(baseAddress[readIndex])
                            writePtr = writePtr.advanced(by: 1)
                            readIndex += strideHidden
                        }
                    }
                }
                result.append(conversionScratch)
            }
            return result
        }
    }

    private func extractHiddenStatesFallback(from multiArray: MLMultiArray, sequenceLength: Int, hiddenDim: Int) throws -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(sequenceLength)
        for seqIdx in 0..<sequenceLength {
            var row: [Float] = []
            row.reserveCapacity(hiddenDim)
            for dimIdx in 0..<hiddenDim {
                let index = [0, seqIdx, dimIdx] as [NSNumber]
                let value = multiArray[index].floatValue
                row.append(value)
            }
            result.append(row)
        }
        return result
    }
    
    private static func compileIfNeeded(at url: URL) throws -> URL {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "mlmodelc":
            return url
        case "mlmodel", "mlpackage":
            return try MLModel.compileModel(at: url)
        default:
            return url
        }
    }

    private static func locateBundledModel(for computeUnits: MLComputeUnits) throws -> URL {
        if computeUnits == .all,
           let aneURL = Bundle.module.url(
                forResource: "GLiNER2EncoderANE",
                withExtension: "mlpackage",
                subdirectory: "Resources"
            ) {
            return aneURL
        }
        if let baseURL = Bundle.module.url(
            forResource: "GLiNER2Encoder",
            withExtension: "mlpackage",
            subdirectory: "Resources"
        ) {
            if computeUnits == .all {
                print("[GLiNEREncoder] Warning: GLiNER2EncoderANE.mlpackage not found; falling back to standard encoder on CPU/GPU path")
            }
            return baseURL
        }
        throw GLiNERError.modelNotFound(
            "GLiNER2Encoder(.ANE).mlpackage not found in bundle. Run Scripts/convert_to_coreml.py to generate it."
        )
    }

    private static func resolveOutputFeatureName(for model: MLModel) -> String {
        let outputs = model.modelDescription.outputDescriptionsByName
        if outputs["hidden_states"] != nil {
            return "hidden_states"
        }
        if let firstKey = outputs.keys.first {
            return firstKey
        }
        return "hidden_states"
    }
}

/// Errors that can occur in GLiNER operations
public enum GLiNERError: LocalizedError {
    case modelNotFound(String)
    case invalidInput(String)
    case invalidOutput(String)
    case tokenizerError(String)
    case encodingError(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let message): return "Model not found: \(message)"
        case .invalidInput(let message): return "Invalid input: \(message)"
        case .invalidOutput(let message): return "Invalid output: \(message)"
        case .tokenizerError(let message): return "Tokenizer error: \(message)"
        case .encodingError(let message): return "Encoding error: \(message)"
        }
    }
}

@available(macOS 10.13, iOS 11.0, tvOS 11.0, watchOS 4.0, *)
extension GLiNEREncoder: @unchecked Sendable {}

