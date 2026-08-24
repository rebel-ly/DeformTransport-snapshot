import hashlib
import inspect
import json
import sys
from pathlib import Path

import wan.modules.trajectory as trajectory

p = Path(inspect.getfile(trajectory)).resolve()
overlay = '/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z/overlay'
wan_move = '/workspace/Wan-Move'
result = {
    'IMPORT_PROBE_COMPLETED': True,
    'trajectory_name': trajectory.__name__,
    'trajectory_package': trajectory.__package__,
    'trajectory_file': trajectory.__file__,
    'trajectory_resolved_file': str(p),
    'trajectory_exists': p.exists(),
    'trajectory_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'relevant_sys_path': [x for x in sys.path if x in (overlay, wan_move)],
}
print(json.dumps(result, sort_keys=True), flush=True)
