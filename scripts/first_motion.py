import mujoco as mj
import mujoco.viewer as viewer
from so101_dual import load, arm_ids, READY_Q

model, data = load()

with viewer.launch_passive(model, data) as v:
    # q = READY_Q.copy()
    while v.is_running():
        mj.mj_step(model, data)
        v.sync()