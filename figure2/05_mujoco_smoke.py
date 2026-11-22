
import mujoco

# 最小真实物理仿真：一个沿 X 轴滑动的物体
xml = """
<mujoco model="P1A_smoke">
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="object_A" pos="0 0 0">
      <joint name="slide_A"
             type="slide"
             axis="1 0 0"/>
      <geom type="box"
            size="0.05 0.05 0.05"
            mass="1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="force_A"
           joint="slide_A"
           gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

initial_x = float(data.qpos[0])

# 在一个滑动方向施加恒定驱动力
data.ctrl[0] = 1.0

for _ in range(500):
    mujoco.mj_step(model, data)

final_x = float(data.qpos[0])

print("MuJoCo version:", mujoco.__version__)
print("Simulation time:", round(data.time, 4))
print("Initial X:", initial_x)
print("Final X:", round(final_x, 6))

assert abs(data.time - 1.0) < 1e-8
assert final_x > initial_x

print("P1-A PASS")
