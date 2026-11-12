"""Known radius-one state controls, not a new optimizer or attack."""
import torch

STATES=('raw','unit-first-frozen','unit-continuous')
STATE_CONTROL_SPEC={
    'raw':'Original unprojected raw proposal; no state transform.',
    'unit-first-frozen':'Per layer/sample: accept first nonzero normalizable raw proposal divided by its FP32 norm, then retain state while all proposals/gradients/draws still execute.',
    'unit-continuous':'Per selected layer/sample: normalize each accepted raw proposal by its FP32 norm; initial true zero delays initialization; subsequent accepted true zero is undefined.',
    'unit_radius':1,'unit_norm':'Author FP32 reshape[B,-1].norm(dim=1); direct division, no epsilon or fallback.',
    'raw_proposal_step':1,'interpretation':'Known unit-norm controls. First-frozen versus raw changes rescaling and subsequent assignments; continuous versus first-frozen is the total effect of this projected gradient rule, not an isolated direction-diversity effect.'}


def validate_state_control(state,update_mode):
    if state not in STATES:raise ValueError('Unknown FTM state control')
    if state!='raw' and update_mode!='full':raise ValueError('Unit-state controls require ftm-update-mode full; cannot combine with first-nonzero-only')
    return state


def state_profile(profile,state,update_mode):
    validate_state_control(state,update_mode)
    if state=='raw':return profile
    suffix='-update-'+update_mode
    if not profile.endswith(suffix):raise ValueError('Unexpected FTM profile suffix')
    return profile[:-len(suffix)]+'-state-'+state+suffix


class StateTransformError(ValueError):
    def __init__(self,kind,step,layer,indices):
        self.details={'kind':kind,'step':step,'layer':layer,'sample_indices':indices,
            'scope':'Known unit-state transform domain; not an original raw FTM error'}
        super().__init__(f'FTM unit-state {kind}; step={step}, layer={layer}, samples={indices}')


class UnitStateControl:
    """Only initialization masks persist; old/new tensors remain caller-owned."""
    def __init__(self,state):
        validate_state_control(state,'full')
        if state=='raw':raise ValueError('Raw state uses state_transform=None')
        self.state=state;self.initialized={}
    def __call__(self,step,name,old,proposal):
        if old.dtype!=torch.float32 or proposal.dtype!=torch.float32:raise ValueError('Unit-state normalization requires FP32 tensors')
        with torch.no_grad():
            initialized=self.initialized.get(name)
            if initialized is None:initialized=torch.zeros(old.shape[0],dtype=torch.bool,device=old.device)
            accept=~initialized if self.state=='unit-first-frozen' else torch.ones_like(initialized)
            result=old.detach().clone()
            indices=torch.nonzero(accept,as_tuple=False).flatten()
            if indices.numel():
                candidate=proposal.detach()[indices]
                flat=candidate.reshape(candidate.shape[0],-1);norm=flat.norm(dim=1)
                nonzero=(flat!=0).any(1)
                def reject(mask,kind):
                    if mask.any():raise StateTransformError(kind,step,name,indices[mask].tolist())
                reject(~torch.isfinite(norm),'nonfinite FP32 proposal norm')
                reject((norm==0)&nonzero,'FP32 norm underflow with nonzero coordinates')
                true_zero=(norm==0)&~nonzero
                reject(true_zero&initialized[indices],'finite degenerate zero accepted after initialization')
                usable=~true_zero
                # True initial zero is accepted as zero but does not initialize.
                normalized=candidate.clone()
                if usable.any():
                    normalized[usable]=candidate[usable]/norm[usable].view(-1,*([1]*(proposal.ndim-1)))
                result[indices]=normalized
                updated=initialized.clone();updated[indices[usable]]=True
                self.initialized[name]=updated
            else:self.initialized[name]=initialized
            return result.detach()
