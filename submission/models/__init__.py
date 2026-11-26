"""Actor code: everything the submitted agent runs at inference.

Hard rule: nothing in this package may import from `training/` or `evaluation/`.

Import from the submodules (`models.actor`, `models.config`, ...). This file stays empty of imports so
environment worker processes can use `models.obs` without loading PyTorch.
"""
