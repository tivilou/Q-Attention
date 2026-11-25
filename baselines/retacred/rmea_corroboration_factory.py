"""Full-run interface only; scientific forward is the exact screened plugin."""
from .rmea_corroboration import CorroborationHead
from .rmea_matched_routers import initialize_head as native_initialize,validate_initialized_head as native_validate

def initialize_head(kind,initial,cfg):
    return CorroborationHead(kind,native_initialize('real12',initial,cfg))

def validate_initialized_head(model,kind,initial,cfg):
    schema=cfg['parameter_schemas'][kind]
    if set(model.state_dict())!=set(schema):raise ValueError('Corroboration wrapper schema differs')
    mapped={k.removeprefix('native.'):v for k,v in schema.items()}
    native_validate(model.native,'real12',initial,dict(cfg,parameter_schemas={'real12':mapped}))
