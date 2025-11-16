import Foundation

enum TestPaths {
    static func projectRoot(file: StaticString = #filePath) -> URL {
        let initial = URL(fileURLWithPath: String(describing: file))
        let components = initial.pathComponents
        guard let repoIndex = components.lastIndex(of: "GlinerML") else {
            fatalError("Unable to locate repository root from \(initial.path)")
        }
        var root = URL(fileURLWithPath: "/")
        if repoIndex >= 1 {
            for component in components[1...repoIndex] {
                root.appendPathComponent(component)
            }
        }
        #if DEBUG
        // Useful when running under DerivedData to ensure we resolve to the workspace checkout.
        print("[TestPaths] Resolved repo root: \(root.path)")
        #endif
        return root
    }

    static func glinerManifestURL(file: StaticString = #filePath) -> URL {
        return projectRoot(file: file)
            .appendingPathComponent("CoreMLArtifacts")
            .appendingPathComponent("export_manifest.json")
    }
}
