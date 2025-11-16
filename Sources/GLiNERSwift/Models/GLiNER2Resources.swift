import Foundation

/// Resolves GLiNER2 Core ML artifacts and tokenizer assets from an export manifest.
struct GLiNER2Resources {
    let manifest: GLiNERManifest
    let resolved: GLiNERManifest.ResolvedResources
    let tokenizerURL: URL

    var encoderURL: URL { resolved.encoderURL }
    var spanRepresentationURL: URL { resolved.spanRepURL }
    var classifierURL: URL { resolved.classifierURL }
    var countPredictorURL: URL { resolved.countPredictorURL }
    var countEmbedURL: URL { resolved.countEmbedURL }

    init(manifestURL: URL) throws {
        let loadedManifest = try GLiNERManifest.load(from: manifestURL)
        let resolvedResources = try loadedManifest.resolveResources(relativeTo: manifestURL)
        guard let tokenizerDirectory = resolvedResources.tokenizerURL else {
            throw GLiNERError.modelNotFound("Manifest at \(manifestURL.path) does not include a tokenizer_dir entry")
        }
        self.manifest = loadedManifest
        self.resolved = resolvedResources
        self.tokenizerURL = tokenizerDirectory
    }
}
