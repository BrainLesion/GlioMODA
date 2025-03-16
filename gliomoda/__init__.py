import os
from gliomoda.constants import NNUNET_ENV_VARS

# suppress nnUNet warnings by setting vars to dummy values (if not already set)
for env_var in NNUNET_ENV_VARS:
    if env_var not in os.environ:
        os.environ[env_var] = ""


from gliomoda.inferer import Inferer
