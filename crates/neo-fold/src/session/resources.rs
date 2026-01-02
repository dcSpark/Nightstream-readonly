use neo_math::F;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::witness::LutTableSpec;
use std::collections::HashMap;
use std::iter;

/// Declarative resource configuration for shared-CPU-bus runs (Twist + Shout).
///
/// This is a thin convenience wrapper around the maps consumed by
/// `neo_memory::builder::build_shard_witness_shared_cpu_bus*`.
#[derive(Clone, Debug, Default)]
pub struct SharedBusResources {
    pub mem_layouts: HashMap<u32, PlainMemLayout>,
    pub initial_mem: HashMap<(u32, u64), F>,
    pub lut_tables: HashMap<u32, LutTable<F>>,
    pub lut_table_specs: HashMap<u32, LutTableSpec>,
}

impl SharedBusResources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn twist(&mut self, twist_id: u32) -> TwistResource<'_> {
        TwistResource { resources: self, twist_id }
    }

    pub fn shout(&mut self, table_id: u32) -> ShoutResource<'_> {
        ShoutResource { resources: self, table_id }
    }

    /// Convenience: set a small binary (n_side=2) 1D table (d=1) with `k = content.len()`.
    pub fn set_binary_table(&mut self, table_id: u32, content: Vec<F>) {
        self.shout(table_id).binary_table(content);
    }

    /// Convenience: set a small 1D memory (d=1) under bit addressing (n_side=2).
    ///
    /// This does not set any initial memory; use `twist(id).init(...)` for that.
    pub fn set_binary_mem_layout(&mut self, twist_id: u32, k: usize) {
        self.twist(twist_id).layout(PlainMemLayout { k, d: 1, n_side: 2 });
    }

    /// Remove any explicit table when switching a table_id to an implicit spec (or vice-versa).
    fn clear_shout_conflicts(&mut self, table_id: u32, keep_spec: bool) {
        if keep_spec {
            self.lut_tables.remove(&table_id);
        } else {
            self.lut_table_specs.remove(&table_id);
        }
    }
}

pub struct TwistResource<'a> {
    resources: &'a mut SharedBusResources,
    twist_id: u32,
}

impl TwistResource<'_> {
    pub fn layout(self, layout: PlainMemLayout) -> Self {
        self.resources.mem_layouts.insert(self.twist_id, layout);
        self
    }

    /// Add sparse initial memory entries for this Twist instance.
    ///
    /// Values of `F::ZERO` are allowed but typically ignored by the underlying builders.
    pub fn init(self, init: impl IntoIterator<Item = (u64, F)>) -> Self {
        for (addr, val) in init {
            self.resources.initial_mem.insert((self.twist_id, addr), val);
        }
        self
    }

    pub fn init_cell(self, addr: u64, val: F) -> Self {
        self.init(iter::once((addr, val)))
    }

    pub fn clear_init(self) -> Self {
        self.resources
            .initial_mem
            .retain(|(mem_id, _addr), _| *mem_id != self.twist_id);
        self
    }
}

pub struct ShoutResource<'a> {
    resources: &'a mut SharedBusResources,
    table_id: u32,
}

impl ShoutResource<'_> {
    pub fn table(self, mut table: LutTable<F>) -> Self {
        table.table_id = self.table_id;
        self.resources.clear_shout_conflicts(self.table_id, /*keep_spec=*/ false);
        self.resources.lut_tables.insert(self.table_id, table);
        self
    }

    pub fn binary_table(self, content: Vec<F>) -> Self {
        let table_id = self.table_id;
        self.table(LutTable {
            table_id,
            k: content.len(),
            d: 1,
            n_side: 2,
            content,
        })
    }

    pub fn spec(self, spec: LutTableSpec) -> Self {
        self.resources.clear_shout_conflicts(self.table_id, /*keep_spec=*/ true);
        self.resources.lut_table_specs.insert(self.table_id, spec);
        self
    }
}
