"""C experiment registry. IDs are stable names in analysis tables."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Intervention:
    name: str
    patch_keys: tuple[str, ...]
    description: str

CONDITIONS = (
    Intervention('original', (), 'Original unmodified V7 generator'),
    Intervention('no_final_log_gate', ('final',), 'Remove log(imprinting_gate) from final expression transform'),
    Intervention('no_grn_imprint_gate', ('grn',), 'Remove explicit GRN-level imprint scaling'),
    Intervention('no_expression_imprint_gates', ('final','inner'), 'Remove final gate plus the inner expression gate factor'),
    Intervention('upstream_only', ('final','inner','grn'), 'Keep upstream region/promoter imprinting; remove explicit expression/GRN gates'),
    Intervention('promoter_equal_final_on', ('promoter',), 'Copy preferred allele promoter state to nonpreferred; keep final gate'),
    Intervention('region_off', ('region_off',), 'Disable imprint-driven region override only'),
    Intervention('promoter_imprint_off', ('promoter_imprint_off',), 'Disable explicit promoter imprint override only'),
    Intervention('region_promoter_only', ('grn','inner','final'), 'Original upstream-only alias'),
    Intervention('promoter_only', ('region_off','grn','inner','final'), 'Retain only promoter-level imprint mechanism'),
    Intervention('region_only', ('promoter_imprint_off','grn','inner','final'), 'Retain only region-level imprint mechanism'),
    Intervention('fully_no_imprinting', ('region_off','promoter_imprint_off','grn','inner','final'), 'Retain labels but disable all five explicit imprint injection paths'),
    Intervention('promoter_equal_final_off', ('promoter','final'), 'Equalize promoter state and remove final gate'),
)
REGISTRY={c.name:c for c in CONDITIONS}
