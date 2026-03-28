//! Genealogical lineage tracking — full ancestry DAG for skills.
//!
//! Every skill maintains a lineage: how it was created, what it evolved from,
//! and what came after. This enables course correction: when planning a new
//! skill version, the planner sees WHY previous versions failed.

use cozo::DataValue;
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::error::Result;
use crate::store::CozoStore;
use crate::types::memory::now_ms;

/// A single entry in the lineage DAG.
#[derive(Debug, Clone)]
pub struct LineageEntry {
    pub child_id: Uuid,
    pub parent_id: Uuid,
    pub relation: String,
    pub generation: i64,
    pub mutation_summary: String,
    pub ts: i64,
}

/// Record a lineage relationship between two skills.
pub fn record_lineage(
    store: &CozoStore,
    child_id: Uuid,
    parent_id: Uuid,
    relation: &str,
    mutation_summary: &str,
) -> Result<()> {
    // Determine generation: parent's max generation + 1
    let parent_gen = get_max_generation(store, parent_id)?;
    let generation = parent_gen + 1;

    let mut params = BTreeMap::new();
    params.insert("child_id".into(), DataValue::from(child_id.to_string()));
    params.insert("parent_id".into(), DataValue::from(parent_id.to_string()));
    params.insert("relation".into(), DataValue::from(relation));
    params.insert("generation".into(), DataValue::from(generation));
    params.insert("mutation_summary".into(), DataValue::from(mutation_summary));
    params.insert("ts".into(), DataValue::from(now_ms()));

    store.run_script(
        concat!(
            "?[child_id, parent_id, relation, generation, mutation_summary, ts] <- ",
            "[[$child_id, $parent_id, $relation, $generation, $mutation_summary, $ts]] ",
            ":put lineage {child_id, parent_id => relation, generation, mutation_summary, ts}",
        ),
        params,
    )?;

    Ok(())
}

/// Get the full ancestry of a skill (walking up the DAG).
pub fn get_full_ancestry(store: &CozoStore, skill_id: Uuid, max_depth: usize) -> Result<Vec<LineageEntry>> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));
    params.insert("max_depth".into(), DataValue::from(max_depth as i64));

    let result = store.run_query(
        r#"ancestors[c, p, r, g, m, t] :=
            *lineage{child_id: c, parent_id: p, relation: r, generation: g, mutation_summary: m, ts: t},
            c = to_uuid($skill_id)
        ancestors[c2, p2, r2, g2, m2, t2] :=
            ancestors[_, prev_parent, _, _, _, _],
            *lineage{child_id: c2, parent_id: p2, relation: r2, generation: g2, mutation_summary: m2, ts: t2},
            c2 = prev_parent,
            g2 <= $max_depth
        ?[child_id, parent_id, relation, generation, mutation_summary, ts] :=
            ancestors[child_id, parent_id, relation, generation, mutation_summary, ts]
        :sort generation"#,
        params,
    )?;

    result.rows.iter().map(|row| parse_lineage_row(row)).collect()
}

/// Get all descendants of a skill (walking down the DAG).
pub fn get_descendants(store: &CozoStore, skill_id: Uuid) -> Result<Vec<LineageEntry>> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));

    let result = store.run_query(
        r#"descendants[c, p, r, g, m, t] :=
            *lineage{child_id: c, parent_id: p, relation: r, generation: g, mutation_summary: m, ts: t},
            p = to_uuid($skill_id)
        descendants[c2, p2, r2, g2, m2, t2] :=
            descendants[prev_child, _, _, _, _, _],
            *lineage{child_id: c2, parent_id: p2, relation: r2, generation: g2, mutation_summary: m2, ts: t2},
            p2 = prev_child
        ?[child_id, parent_id, relation, generation, mutation_summary, ts] :=
            descendants[child_id, parent_id, relation, generation, mutation_summary, ts]
        :sort generation"#,
        params,
    )?;

    result.rows.iter().map(|row| parse_lineage_row(row)).collect()
}

fn get_max_generation(store: &CozoStore, skill_id: Uuid) -> Result<i64> {
    let mut params = BTreeMap::new();
    params.insert("skill_id".into(), DataValue::from(skill_id.to_string()));

    let result = store.run_query(
        r#"?[max_gen] :=
            *lineage{child_id, generation},
            child_id = to_uuid($skill_id),
            max_gen = max(generation)
        ?[max_gen] := max_gen = 0"#,
        params,
    )?;

    Ok(result.rows.first().and_then(|r| r[0].get_int()).unwrap_or(0))
}

fn parse_lineage_row(row: &[DataValue]) -> Result<LineageEntry> {
    let child_id = crate::store::parse_uuid_pub(&row[0])?;
    let parent_id = crate::store::parse_uuid_pub(&row[1])?;
    let relation = row[2].get_str().unwrap_or("").to_string();
    let generation = row[3].get_int().unwrap_or(0);
    let mutation_summary = row[4].get_str().unwrap_or("").to_string();
    let ts = row[5].get_int().unwrap_or(0);

    Ok(LineageEntry {
        child_id,
        parent_id,
        relation,
        generation,
        mutation_summary,
        ts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get_lineage() {
        let store = CozoStore::open_mem(4).unwrap();
        let parent_id = Uuid::now_v7();
        let child_id = Uuid::now_v7();

        record_lineage(&store, child_id, parent_id, "evolved_from", "added error handling").unwrap();

        let ancestry = get_full_ancestry(&store, child_id, 10).unwrap();
        assert_eq!(ancestry.len(), 1);
        assert_eq!(ancestry[0].parent_id, parent_id);
        assert_eq!(ancestry[0].relation, "evolved_from");
    }

    #[test]
    fn test_get_descendants() {
        let store = CozoStore::open_mem(4).unwrap();
        let root = Uuid::now_v7();
        let child = Uuid::now_v7();
        let grandchild = Uuid::now_v7();

        record_lineage(&store, child, root, "evolved_from", "v2").unwrap();
        record_lineage(&store, grandchild, child, "evolved_from", "v3").unwrap();

        let descendants = get_descendants(&store, root).unwrap();
        assert!(descendants.len() >= 1);
    }
}
