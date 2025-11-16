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
        
        // Flatten all entities
        var allEntities = entitiesPerChunk.flatMap { $0 }
        
        // Sort by score (descending)
        allEntities.sort { $0.score > $1.score }
        
        // Deduplicate overlapping entities
        return deduplicateEntities(allEntities)
    }
    
    /// Remove duplicate entities from overlapping chunks
    private func deduplicateEntities(_ entities: [Entity]) -> [Entity] {
        var selected: [Entity] = []
        
        for entity in entities {
            // Check if this entity overlaps significantly with any already selected
            let isDuplicate = selected.contains { existing in
                guard existing.label == entity.label else {
                    return false
                }
                
                // Check for significant overlap
                return hasSignificantOverlap(entity1: existing, entity2: entity)
            }
            
            if !isDuplicate {
                selected.append(entity)
            }
        }
        
        return selected
    }
    
    /// Check if two entities have significant overlap
    private func hasSignificantOverlap(entity1: Entity, entity2: Entity) -> Bool {
        // Check text similarity (for entities from different chunks that represent the same thing)
        if entity1.text.lowercased() == entity2.text.lowercased() {
            return true
        }
        
        // Check range overlap using start/end
        let start1 = entity1.start
        let end1 = entity1.end
        let start2 = entity2.start
        let end2 = entity2.end
        
        // Check if ranges overlap
        return start1 < end2 && start2 < end1
    }
}
