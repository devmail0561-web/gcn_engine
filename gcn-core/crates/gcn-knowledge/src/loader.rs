use std::collections::HashMap;
use std::path::Path;

use crate::taxonomy::Taxonomy;
use crate::KnowledgeError;

pub fn load_taxonomy(path: &Path) -> Result<Taxonomy, KnowledgeError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| KnowledgeError::Io(e, path.to_path_buf()))?;
    let taxonomy: Taxonomy = yaml_serde::from_str(&content)
        .map_err(|e| KnowledgeError::Yaml(e, path.to_path_buf()))?;
    Ok(taxonomy)
}

pub fn load_all_taxonomies(dir: &Path) -> Result<HashMap<String, Taxonomy>, KnowledgeError> {
    let mut taxonomies = HashMap::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| KnowledgeError::Io(e, dir.to_path_buf()))?;
    for entry in entries {
        let entry = entry.map_err(|e| KnowledgeError::Io(e, dir.to_path_buf()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let taxonomy = load_taxonomy(&path)?;
            taxonomies.insert(taxonomy.taxonomy.clone(), taxonomy);
        }
    }
    Ok(taxonomies)
}
