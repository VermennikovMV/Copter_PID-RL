# WORKSPACE_ONLY recording sources

These original cmd/PowerShell sources were used in the full project. They are
review copies; required run/progress data, guards, WSL distributions and external
Gazebo stack are not all included. External home strings are redacted where
needed; original/export SHA are separate. Do not execute this archive as a clean
clone training recipe. Original v21 stopped with an evaluator metadata error;
v21p1 is the completed corrected PPO pilot, and all six candidates failed quality.
v23 is a completed one-arm PPO pilot; all three combined admissions failed.

Original workspace commands: show_joint_memory20.cmd opens real Gazebo;
train_joint_memory21p1.cmd and train_joint_heading23.cmd start genuine PyBullet
fine-tuning through official preflight. Training is not performed inside Gazebo.
The show_joint_training_native_v*.cmd viewers read actual progress without HTTP.
Archived data displays completed training; it is not live learning. For a live
video the user must start a fresh training run and point the matching viewer at
that new run. A fresh model inherits no measured applicability cells.

The v20 GUI integration is separate from the nine-new/eleven-shared measured
matrix. No automatic video recording or new GUI process was launched here.

V24 original workspace: train_joint_initial24.cmd for true guarded fine-tuning; show_joint_training_native_v24.cmd for its actual counters. Its completed candidates all failed full admission and never flew in Gazebo. These copies retain WORKSPACE_ONLY scope.
