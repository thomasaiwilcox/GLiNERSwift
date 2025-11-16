import Foundation

public final class EntityReporter {
    private var entitiesByLabel: [String: Set<String>] = [:]

    public init() {}

    public func record(_ entities: [Entity]) {
        guard !entities.isEmpty else { return }
        for entity in entities {
            var values = entitiesByLabel[entity.label, default: []]
            values.insert(entity.text)
            entitiesByLabel[entity.label] = values
        }
    }

    public func groupedEntities() -> [String: [String]] {
        var result: [String: [String]] = [:]
        for (label, values) in entitiesByLabel {
            result[label] = values.sorted()
        }
        return result
    }

    public func printSummary() {
        let groups = groupedEntities()
        print("\n=== Entity Details ===")
        guard !groups.isEmpty else {
            print("No entities extracted during the benchmark.")
            return
        }
        for label in groups.keys.sorted() {
            print("\(label):")
            for value in groups[label, default: []] {
                print("  - \(value)")
            }
        }
    }

    public var isEmpty: Bool {
        entitiesByLabel.isEmpty
    }
}
