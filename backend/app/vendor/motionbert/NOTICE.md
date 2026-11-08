# MotionBERT

Upstream: https://github.com/Walter0807/MotionBERT

Revision: `705d3a95354db8bdb696b3492e47a3b5537174ff`.

`DSTformer.py` and `drop.py` are copied from `lib/model/` at that revision.
The only code change is the relative import `from .drop import DropPath`.
The upstream Apache 2.0 license is included in `LICENSE`. Existing copyright
notices in the files are preserved. No training or mesh code is bundled.

Zhu et al., *MotionBERT: A Unified Perspective on Learning Human Motion
Representations*, ICCV 2023. https://arxiv.org/abs/2210.06551

Weights are downloaded separately from the author's Hugging Face repository,
revision `370a9196aa3c89198b134c82476143b01c0fb32c`, checkpoint
`FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`. The download script verifies
its SHA-256. Code licensing alone does not establish conditions for every
training dataset or future distribution of weights; consult the upstream
model card and Human3.6M conditions before distributing a trained product.
