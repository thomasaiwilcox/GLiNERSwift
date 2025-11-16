import CoreML

extension MLMultiArray {
    func withUnsafeMutableFloat32Buffer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) throws -> R {
        guard dataType == .float32 else {
            throw GLiNERError.invalidOutput("Expected float32 buffer, got \(dataType)")
        }
        let pointer = dataPointer.bindMemory(to: Float.self, capacity: count)
        let buffer = UnsafeMutableBufferPointer(start: pointer, count: count)
        return try body(buffer)
    }

    func withUnsafeFloat32Buffer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) throws -> R {
        guard dataType == .float32 else {
            throw GLiNERError.invalidOutput("Expected float32 buffer, got \(dataType)")
        }
        let pointer = dataPointer.bindMemory(to: Float.self, capacity: count)
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        return try body(buffer)
    }

    func withUnsafeFloat16Buffer<R>(_ body: (UnsafeBufferPointer<UInt16>) throws -> R) throws -> R {
        guard dataType == .float16 else {
            throw GLiNERError.invalidOutput("Expected float16 buffer, got \(dataType)")
        }
        let pointer = dataPointer.bindMemory(to: UInt16.self, capacity: count)
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        return try body(buffer)
    }
}
