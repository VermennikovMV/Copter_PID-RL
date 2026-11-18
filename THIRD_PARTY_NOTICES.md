# Third-party notices

The project is licensed under the [MIT License](LICENSE), except the files listed under
"Included in this repository".

## Included in this repository

| Path | Origin | License |
|---|---|---|
| `simulation/models/iris/` | Iris model of [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo) | see `simulation/models/iris/LICENSE.txt` |

## Used, not included

| Component | License | Use |
|---|---|---|
| [ArduPilot](https://github.com/ArduPilot/ardupilot) / [ardupilot-ROSMeetUp](https://github.com/VermennikovMV/ardupilot-ROSMeetUp) | GPL-3.0-or-later | SITL autopilot |
| [Gazebo Harmonic](https://gazebosim.org) (gz-sim) | Apache-2.0 | simulator |
| [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo) | LGPL-3.0 | SITL ↔ Gazebo bridge plugin |
| NumPy, SciPy | BSD-3-Clause | |
| PyBullet | zlib | training simulator |
| PyTorch | BSD-3-Clause | |
| Stable-Baselines3 | MIT | PPO |
| Gymnasium | MIT | environment API |
| pymavlink | LGPL-3.0 | MAVLink |

Python packages are installed from `requirements.txt`; their license texts ship with the packages.
