# TwinWorld-P0: Seeing Computer Vision while Standing on the Shoulders of Giants

**Research Objective**: To verify that worlds with different underlying hidden mechanisms are indistinguishable under perfectly matched passive trajectories, whereas object B exhibits distinct responses when the same external force is applied to object A.

## Execution Environment
- Windows PowerShell / Python 3.11
- Dependencies: numpy, matplotlib only. No GPU, MuJoCo, PyTorch or dataset download is required.

## Sequential Execution

Open this folder in VS Code and select PowerShell as the terminal.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\01_generate.py
.\.venv\Scripts\python.exe scripts\02_validate.py
.\.venv\Scripts\python.exe scripts\03_baseline.py
.\.venv\Scripts\python.exe scripts\04_plot.py
```

If `py -3.11` is unavailable, confirm Python 3.11 is installed; inspect local Python versions with `py --list`.

**Outputs**:

- `data/{train,val,test}_public.npz`: observable inputs for models, containing both passive and interventional trajectories without hidden labels.
- `data/{train,val,test}_private_labels.npz`: mechanism labels reserved exclusively for evaluation; must not be read during model inference.
- `data/{train,val,test}_private_params.json`: ground-truth generator parameters, reserved for auditing only.
- `outputs/p0_baselines.json`: results for passive and active diagnosis.
- `outputs/p0_twinworld_comparison.png`: trajectory visualization.

## Mandatory Checks for P0 Conclusions
- All three unit tests in the unittest script pass.
- For every split: `passive_max_abs_diff = 0` and `matched_action_diff = 0`.
- Object B shows clearly different motion under the two mechanisms after intervention (this is separate from the sensor noise tolerance defined in P1).
- Passive and active oracle diagnosis outcomes are determined by the implementation; a value close to 1.000 should not be treated as a novel contribution of the paper.
- `passive chance = 0.500`
- Do not modify the original test labels or metric definitions; preserve the full terminal log.

## Core Experimental Logic
- `label = 1`: The two objects have hidden spring-damper coupling; the coupling force depends on relative displacement and velocity.
- `label = 0`: The two objects are independent, though they may produce identical motion in the passive regime.
- Initial displacements, velocities and friction terms of the two worlds are perfectly matched. The resting length of the spring equals the object separation, so the spring exerts no force during passive motion.
- External force is applied only to object A. Under the coupled mechanism, this force propagates through the spring to affect object B; under the independent mechanism, object B remains unaffected.
- No cues revealing the spring, hidden labels or generator ground truth are present in the model inputs.
