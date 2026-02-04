use crate::PiCcsError;

/// Deterministically split a Shout instance's `ell_addr` (per lane) across multiple committed mats
/// so each mat's Shout bus tail fits within the witness width `m` without overlapping `m_in`.
///
/// Each page encodes `page_ell_addr` address columns per lane, plus the canonical `[has_lookup, val]`.
/// The returned vector contains the per-page `page_ell_addr` values (in order).
pub(crate) fn plan_shout_addr_pages(
    m: usize,
    m_in: usize,
    steps: usize,
    ell_addr: usize,
    lanes: usize,
) -> Result<Vec<usize>, PiCcsError> {
    if steps == 0 {
        return Err(PiCcsError::InvalidInput(
            "Shout paging requires steps>=1".into(),
        ));
    }
    if m_in > m {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout paging requires m_in<=m (m_in={m_in}, m={m})"
        )));
    }
    let lanes = lanes.max(1);
    let avail = m - m_in;

    // `BusLayout` requires `bus_base >= m_in`, i.e. `bus_cols*steps <= m - m_in`.
    let max_bus_cols_total = avail / steps;
    let per_lane_capacity = max_bus_cols_total / lanes;
    if per_lane_capacity < 3 {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout paging: insufficient capacity for 1 lane (need >=3 cols per lane for [addr_bits>=1,has_lookup,val], have per_lane_capacity={per_lane_capacity}; m={m}, m_in={m_in}, steps={steps}, lanes={lanes})"
        )));
    }
    let max_addr_cols_per_page = per_lane_capacity - 2;

    if ell_addr == 0 {
        return Err(PiCcsError::InvalidInput(
            "Shout paging: ell_addr must be >= 1".into(),
        ));
    }

    let mut out = Vec::new();
    let mut remaining = ell_addr;
    while remaining > 0 {
        let take = remaining.min(max_addr_cols_per_page);
        out.push(take);
        remaining -= take;
    }
    Ok(out)
}

