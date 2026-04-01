#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${SUDO_USER:-${USER:-soccz}}"

echo "[gpu-fix] checking NVIDIA runtime"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[gpu-fix] nvidia-smi not found"
fi

echo "[gpu-fix] ensuring /dev/nvidia* nodes exist"
mkdir -p /dev/nvidia-caps

if [ ! -e /dev/nvidiactl ]; then
  mknod -m 666 /dev/nvidiactl c 195 255
fi
if [ ! -e /dev/nvidia0 ]; then
  mknod -m 666 /dev/nvidia0 c 195 0
fi
if [ ! -e /dev/nvidia-modeset ]; then
  mknod -m 666 /dev/nvidia-modeset c 195 254
fi

UVM_MAJOR="$(awk '$2 == "nvidia-uvm" {print $1}' /proc/devices | tail -n1)"
if [ -n "${UVM_MAJOR:-}" ]; then
  if [ ! -e /dev/nvidia-uvm ]; then
    mknod -m 666 /dev/nvidia-uvm c "$UVM_MAJOR" 0
  fi
  if [ ! -e /dev/nvidia-uvm-tools ]; then
    mknod -m 666 /dev/nvidia-uvm-tools c "$UVM_MAJOR" 1
  fi
fi

echo "[gpu-fix] adding ${USER_NAME} to video/render groups if they exist"
if getent group video >/dev/null 2>&1; then
  usermod -aG video "$USER_NAME" || true
fi
if getent group render >/dev/null 2>&1; then
  usermod -aG render "$USER_NAME" || true
fi

echo "[gpu-fix] final device state"
ls -l /dev/nvidia* || true

echo "[gpu-fix] final nvidia-smi"
nvidia-smi

echo "[gpu-fix] done"
echo "[gpu-fix] if group membership changed, re-login before running training"
