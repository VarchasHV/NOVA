# Training on the GPU laptops

Training runs on laptops with an NVIDIA GPU. The Mac starts, watches, stops and collects the runs
over SSH with [`tools/remote.sh`](../tools/remote.sh). Each laptop trains its own run: PyTorch cannot
use another machine's GPU, and the training code does not split one run across machines.

| Alias | Machine | OS | GPU memory | Planned run |
|---|---|---|---|---|
| `acer` | Acer ALG, i5-13500H | CachyOS (fish shell) | 6 GB | `configs/full.yaml` (submission candidate) |
| `asus` | ASUS TUF A15, Ryzen 7 7445HS | Ubuntu 24.04 | 4 GB | `configs/gru_baseline.yaml` (no-memory ablation) |

If the speed check says the ASUS is faster, swap the two runs.

## 1. Packages on each laptop

**CachyOS**

```bash
sudo pacman -Syu
sudo pacman -S --needed openssh rsync tmux git git-lfs uv libglvnd glib2
sudo systemctl enable --now sshd
sudo ufw allow ssh        # only if ufw is installed
```

**Ubuntu 24.04**

```bash
sudo apt update
sudo apt install -y openssh-server rsync tmux git git-lfs curl libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo systemctl enable --now ssh
sudo ufw allow ssh
```

**NVIDIA driver:** `nvidia-smi` must work and show driver 580 or newer. If it does not: on CachyOS
run `sudo chwd -a`; on Ubuntu pick a driver from `ubuntu-drivers devices` and install it. Reboot
after any kernel or driver update.

**Not needed:** a system CUDA toolkit, cuDNN, conda, a system Python or Docker. `uv sync` installs
Python 3.13, PyTorch 2.15.0 with its CUDA libraries, and ViZDoom 1.3.3.

## 2. Keep the laptops awake

On both laptops:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Undo later with `unmask`. Keep them plugged in, on the performance power mode, with the vents clear
and browsers closed (each has 16 GB RAM).

## 3. SSH from the Mac

Find each laptop's Wi-Fi address with `ip -4 -br addr`. Addresses can change after a reconnect.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/nova_train      # once
```

Add to `~/.ssh/config`:

```
Host acer
  HostName <acer address>
  User unhealthycoder
  IdentityFile ~/.ssh/nova_train
  IdentitiesOnly yes
  ServerAliveInterval 30

Host asus
  HostName <asus address>
  User charlie
  IdentityFile ~/.ssh/nova_train
  IdentitiesOnly yes
  ServerAliveInterval 30
```

Install the key (asks for the laptop's password once), then test:

```bash
ssh-copy-id -i ~/.ssh/nova_train.pub acer
ssh-copy-id -i ~/.ssh/nova_train.pub asus
ssh acer nvidia-smi && ssh asus nvidia-smi
```

## 4. Run training (on the Mac, from the repository root)

```bash
tools/remote.sh setup acer                      # copies code, installs packages, checks speed
tools/remote.sh setup asus
```

The first setup downloads several GB on each laptop. If it is interrupted, run it again. It must end
with `READY`; note the `recommended: --set num_workers=N` line.

Smoke test (up to 400k decisions or 1 hour):

```bash
tools/remote.sh start acer configs/smoke.yaml
tools/remote.sh status acer
tools/remote.sh stop acer
```

Real runs. `H` is the hours left minus about 3 for export and submission:

```bash
tools/remote.sh start acer configs/full.yaml --set num_workers=N --set max_hours=H
tools/remote.sh start asus configs/gru_baseline.yaml --set num_workers=N --set max_hours=H
```

Training keeps running if the Mac sleeps or leaves the network.

| Command | Does |
|---|---|
| `tools/remote.sh status <host>` | GPU load, memory and the last log lines |
| `tools/remote.sh stop <host>` | saves a checkpoint and stops |
| `tools/remote.sh board <host> [port]` | TensorBoard at `http://localhost:16006`; use another port for the second laptop |
| `tools/remote.sh fetch <host>` | copies the runs to `runs/<host>/` on the Mac; do this every few hours as a backup |
| `tools/remote.sh sync <host>` | copies code changes (only while that laptop is idle) |

**Resume:** run the same `start` command again, with the same `--set` options. `max_hours` is the total
for the run, including time already trained.

**Move a run to the other laptop** (for example from the Acer to the ASUS):

```bash
tools/remote.sh fetch acer                      # skip if the Acer is unreachable; use the last copy
tools/remote.sh stop asus
rsync -az runs/acer/full/ asus:nova-train/runs/full/
tools/remote.sh start asus configs/full.yaml --set num_workers=N --set max_hours=H
```

## 5. Export and submit (on the Mac)

```bash
tools/remote.sh fetch acer
uv run python -m training.export --ckpt runs/acer/full/best.pt    # ckpt_latest.pt if there is no best.pt
uv run python validate.py
```

`best.pt` is written by the held-out evaluation that runs every 30 minutes during training. To submit,
the weights must be committed with Git LFS (`brew install git-lfs`); follow
[SUBMISSION.md](SUBMISSION.md). `validate.py` also checks that everything under `submission/` is
committed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No route to host` from VS Code, but Terminal.app connects | macOS System Settings, Privacy & Security, Local Network: allow Visual Studio Code |
| `No route to host` from every app | The Wi-Fi blocks device-to-device traffic: use Tailscale on all three, or a phone hotspot |
| `Connection refused` | SSH server not running on the laptop: `sudo systemctl enable --now ssh` (CachyOS: `sshd`) |
| `Host key verification failed` | Connect once with `ssh <host>` and accept the key |
| `setup` reports `NO CUDA` | Check `nvidia-smi` on the laptop; reboot after a driver update |
| Out of system memory | Lower `--set num_workers=N` |
| Out of GPU memory | Add `--set ppo.minibatches=16` |
| "trained with different model settings" | Use `--run-name new_name`, or `--fresh` to overwrite that run |
