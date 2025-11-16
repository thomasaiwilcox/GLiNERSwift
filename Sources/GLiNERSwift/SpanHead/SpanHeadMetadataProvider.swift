import Foundation

enum SpanHeadMetadataProvider {
    private static var cached: SpanHeadMetadata?
    private static let lock = NSLock()

    static func metadata(bundle: Bundle = .module) throws -> SpanHeadMetadata {
        lock.lock()
        defer { lock.unlock() }
        if let cached {
            return cached
        }
        guard let baseURL = bundle.resourceURL?.appendingPathComponent("SpanHead", isDirectory: true) else {
            throw GLiNERError.encodingError("SpanHead resources directory not found in bundle")
        }
        let metadataURL = baseURL.appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metadataURL)
        let decoded = try JSONDecoder().decode(SpanHeadMetadata.self, from: data)
        cached = decoded
        return decoded
    }
}
