import Foundation

/// Processes chunks from sliding window tokenization and merges results
public class ChunkProcessor {
    private let config: Configuration
    
    public init(config: Configuration = .default) {
        self.config = config
    }
    
    /// Merge entities from overlapping chunks
    /// - Parameter entitiesPerChunk: Array of entity arrays (one per chunk)
    /// - Returns: Merged and deduplicated entities
    public func mergeChunks(_ entitiesPerChunk: [[Entity]]) -> [Entity] {
        guard !entitiesPerChunk.isEmpty else {
            return []
        }
        
        // If only one chunk, return as-is
        if entitiesPerChunk.count == 1 {
            return entitiesPerChunk[0]
        }
        
        // Flatten all entities with estimated capacity
        let estimatedCount = entitiesPerChunk.reduce(0) { $0 + $1.count }
        var allEntities: [Entity] = []
        allEntities.reserveCapacity(estimatedCount)
        
        for entities in entitiesPerChunk {
            allEntities.append(contentsOf: entities)
        }
        
        // Sort by score (descending)
        allEntities.sort { $0.score > $1.score }
        
        // Deduplicate overlapping entities with optimized algorithm
        return deduplicateEntities(allEntities)
    }
    
    /// Remove duplicate entities from overlapping chunks using optimized algorithm
    private func deduplicateEntities(_ entities: [Entity]) -> [Entity] {
        guard !entities.isEmpty else {
            return []
        }
        
        var selected: [Entity] = []
        selected.reserveCapacity(entities.count / 2) // Heuristic pre-allocation
        
        // Group by label for efficient deduplication
        var byLabel: [String: [Entity]] = [:]
        for entity in entities {
            byLabel[entity.label, default: []].append(entity)
        }
        
        // Process each label group independently
        for (_, labelEntities) in byLabel {
            var labelSelected: [Entity] = []
            
            for entity in labelEntities {
                // Check if this entity overlaps significantly with any already selected
                let isDuplicate = labelSelected.contains { existing in
                    hasSignificantOverlapFast(entity1: existing, entity2: entity)
                }
                
                if !isDuplicate {
                    labelSelected.append(entity)
                }
            }
            
            selected.append(contentsOf: labelSelected)
        }
        
        // Re-sort by score to maintain best-first order
        return selected.sorted { $0.score > $1.score }
    }
    
    /// Optimized overlap check with early exit
    @inline(__always)
    private func hasSignificantOverlapFast(entity1: Entity, entity2: Entity) -> Bool {
        // Quick text comparison first (cheapest check)
        if entity1.text.count == entity2.text.count {
            if entity1.text.lowercased() == entity2.text.lowercased() {
                return true
            }
        }
        
        // Range overlap check with early exit
        let start1 = entity1.start
        let end1 = entity1.end
        let start2 = entity2.start
        let end2 = entity2.end
        
        // Check if ranges overlap
        return start1 < end2 && start2 < end1
    }
}
