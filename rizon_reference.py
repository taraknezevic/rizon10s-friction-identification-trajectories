import mujoco
import mujoco.viewer
import pandas as pd
import time
from pathlib import Path

MODEL_PATH = r"C:\m\mujoco_menagerie\flexiv_rizon_10s\FlexivPy\assets\mjmodel.xml"

base_dir = Path.cwd()

#TRAJ_PATH = base_dir / "identif_flexiv" / "rizon10s_gen_traj" / "q7" / "p09" / "v_3.25" / "fourier" / "traj_neg_position.csv"
TRAJ_PATH = base_dir / "valid_flexiv" / "rizon10s_valid_traj" / "q7" / "p03" / "v_1.50" / "fourier" / "traj_valid_neg_position.csv"

print("Tražim file:")
print(TRAJ_PATH)

from pathlib import Path

print(TRAJ_PATH)
print("Postoji:", TRAJ_PATH.exists())
print("Folder postoji:", TRAJ_PATH.parent.exists())

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

traj = pd.read_csv(TRAJ_PATH, header=None).values

print(traj.shape)
print(traj[0])
print(traj[100])
print(traj[-1])

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    for k in range(0, len(traj), 20):

        data.qpos[:7] = traj[k]

        mujoco.mj_forward(model, data)

        print(data.qpos[:7])

        viewer.sync()

        time.sleep(0.05)

    print("Trajectory finished")