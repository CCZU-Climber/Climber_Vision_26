#!/usr/bin/env python3
"""Unified perception camera manager.

Features:
- Query video devices and existing /dev/camera_* links
- Inspect existing udev camera rules
- Generate/apply/refresh udev rules with sudo
- Round-robin preview (single-camera-per-iteration, 10ms)
- Export diagnostic reports (json + txt)

UI mode:  python3 scripts/perception_camera_manager.py ui
CLI mode: python3 scripts/perception_camera_manager.py check
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sentry.yaml"
DEFAULT_RULES_PATH = Path("/etc/udev/rules.d/99-camera-names.rules")
REPORT_DIR = PROJECT_ROOT / "logs"
POLL_INTERVAL_SEC = 0.01

# Left / Right / Back templates (English)
ROLE_TEMPLATES = [
    ("Left", "camera_back_left"),
    ("Right", "camera_back_right"),
    ("Back", "camera_back"),
]


@dataclass
class DeviceInfo:
    path: str
    realpath: str
    id_path: str
    id_vendor_id: str
    id_model_id: str
    id_vendor: str
    id_model: str
    devpath: str
    open_ok: bool
    frame_ok: bool
    width: int
    height: int
    fps: float


@dataclass
class RuleApplyResult:
    ok: bool
    message: str
    fallback_commands: List[str]


def run_cmd(cmd: List[str], timeout: int = 8, capture_output: bool = True) -> subprocess.CompletedProcess:
    if capture_output:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    return subprocess.run(cmd, text=True, timeout=timeout, check=False)


def safe_run_text(cmd: List[str], timeout: int = 8) -> Tuple[bool, str]:
    try:
        proc = run_cmd(cmd, timeout=timeout)
    except Exception as e:
        return False, str(e)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}"
        return False, err
    return True, proc.stdout


def ensure_sudo_ready(auto_prompt: bool = True) -> Tuple[bool, str]:
    proc = run_cmd(["sudo", "-n", "true"], timeout=3)
    if proc.returncode == 0:
        return True, "sudo credential ready"

    err = proc.stderr.strip() or proc.stdout.strip() or "sudo auth required"

    if auto_prompt and sys.stdin is not None and sys.stdin.isatty():
        print("[perception_camera_manager] sudo需要认证，请在当前终端输入密码...", flush=True)
        try:
            auth_proc = run_cmd(["sudo", "-v"], timeout=120, capture_output=False)
        except Exception as e:
            return False, f"sudo认证异常: {e}"

        if auth_proc.returncode == 0:
            ok2 = run_cmd(["sudo", "-n", "true"], timeout=3)
            if ok2.returncode == 0:
                return True, "sudo credential cached"
            err2 = ok2.stderr.strip() or ok2.stdout.strip() or "sudo cache check failed"
            return False, f"sudo认证后校验失败: {err2}"

        return False, f"sudo认证失败，退出码={auth_proc.returncode}"

    hint = (
        "sudo未认证，无法在GUI内交互输入密码。"
        "请在终端先执行 `sudo -v` 完成认证，再回到UI点击应用。"
    )
    return False, f"{hint} 原因: {err}"

def parse_udev_properties(device_path: str) -> Dict[str, str]:
    ok, out = safe_run_text(["udevadm", "info", "--query=property", "--name", device_path], timeout=6)
    if not ok:
        return {}

    props: Dict[str, str] = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        props[k.strip()] = v.strip()
    return props


def parse_kernels_match(device_path: str) -> Optional[str]:
    ok, out = safe_run_text(["udevadm", "info", "-a", "-n", device_path], timeout=8)
    if not ok:
        return None

    for line in out.splitlines():
        line = line.strip()
        if line.startswith('KERNELS=="') and ":1.0" in line:
            return line
    return None


def scan_video_devices(test_open: bool = True) -> List[DeviceInfo]:
    devices = sorted(glob.glob("/dev/video*"))
    infos: List[DeviceInfo] = []

    for dev in devices:
        real = os.path.realpath(dev)
        props = parse_udev_properties(dev)

        open_ok = False
        frame_ok = False
        width = 0
        height = 0
        fps = 0.0

        if test_open:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            open_ok = bool(cap.isOpened())
            if open_ok:
                ret, frame = cap.read()
                frame_ok = bool(ret and frame is not None and frame.size > 0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()

        infos.append(
            DeviceInfo(
                path=dev,
                realpath=real,
                id_path=props.get("ID_PATH", ""),
                id_vendor_id=props.get("ID_VENDOR_ID", ""),
                id_model_id=props.get("ID_MODEL_ID", ""),
                id_vendor=props.get("ID_VENDOR", ""),
                id_model=props.get("ID_MODEL", ""),
                devpath=props.get("DEVPATH", ""),
                open_ok=open_ok,
                frame_ok=frame_ok,
                width=width,
                height=height,
                fps=fps,
            )
        )

    return infos


def list_camera_symlinks() -> Dict[str, str]:
    links = {}
    for path in sorted(glob.glob("/dev/camera*")):
        links[path] = os.path.realpath(path)
    return links


def read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def write_yaml(path: Path, data: Dict) -> Tuple[bool, str]:
    try:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _replace_top_level_yaml_block(original_text: str, key: str, new_block_text: str) -> str:
    lines = original_text.splitlines(keepends=True)
    key_re = re.compile(rf"^([ \t]*){re.escape(key)}\s*:(?:\s*#.*)?\s*$")

    start = -1
    base_indent = 0
    for idx, line in enumerate(lines):
        m = key_re.match(line)
        if not m:
            continue
        indent = m.group(1)
        if indent:
            continue
        start = idx
        base_indent = len(indent)
        break

    block = new_block_text.rstrip("\n") + "\n"

    if start < 0:
        tail = original_text.rstrip("\n")
        if not tail:
            return block
        return tail + "\n\n" + block

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        current_indent = len(line) - len(line.lstrip(" \t"))
        if current_indent <= base_indent:
            end = idx
            break

    before = "".join(lines[:start])
    after = "".join(lines[end:])
    return before + block + after


def write_camera_name_map_only(path: Path, camera_name_map: Dict) -> Tuple[bool, str]:
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        block = yaml.safe_dump(
            {"camera_name_map": camera_name_map},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        updated = _replace_top_level_yaml_block(original, "camera_name_map", block)
        path.write_text(updated, encoding="utf-8")
        return True, "ok"
    except Exception as e:
        return False, str(e)


def get_camera_name_map(config_path: Path) -> Dict:
    data = read_yaml(config_path)
    cmap = data.get("camera_name_map", {})
    return cmap if isinstance(cmap, dict) else {}


def get_existing_rules_text(rules_path: Path = DEFAULT_RULES_PATH) -> str:
    try:
        return rules_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def generate_rules_text(bindings: Dict[str, str], comment: str = "Generated by perception_camera_manager") -> Tuple[str, List[str]]:
    lines = [
        "# USB camera fixed device-name rules",
        f"# {comment}",
        f"# Generated at {datetime.now().isoformat()}",
        "",
    ]
    warnings: List[str] = []

    for role, selected_device in bindings.items():
        selected_device = selected_device.strip()
        if not selected_device:
            continue

        dev = os.path.realpath(selected_device)
        props = parse_udev_properties(dev)

        match_expr = ""
        if props.get("ID_PATH"):
            match_expr = f'ENV{{ID_PATH}}=="{props["ID_PATH"]}"'
        else:
            kernels = parse_kernels_match(dev)
            if kernels:
                match_expr = kernels
            else:
                warnings.append(f"{role}: cannot determine stable match key from {selected_device}")
                continue

        line = (
            "SUBSYSTEM==\"video4linux\", "
            f"{match_expr}, "
            "ATTR{index}==\"0\", "
            f"SYMLINK+=\"{role}\", "
            "GROUP=\"video\", MODE=\"0666\""
        )
        lines.append(line)

    if len(lines) == 4:
        warnings.append("No valid bindings selected, rule content is empty")

    lines.append("")
    return "\n".join(lines), warnings


def apply_rules_text(rules_text: str, rules_path: Path = DEFAULT_RULES_PATH) -> RuleApplyResult:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = rules_path.with_name(rules_path.name + f".bak.{ts}")

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".rules") as tmp:
        tmp.write(rules_text)
        tmp_path = tmp.name

    fallback: List[str] = []
    try:
        ok_sudo, sudo_msg = ensure_sudo_ready(auto_prompt=True)
        if not ok_sudo:
            fallback.extend([
                "sudo -v",
                f"sudo cp {shlex.quote(tmp_path)} {shlex.quote(str(rules_path))}",
                "sudo udevadm control --reload-rules",
                "sudo udevadm trigger",
            ])
            return RuleApplyResult(False, sudo_msg, fallback)

        if rules_path.exists():
            cmd = ["sudo", "-n", "cp", str(rules_path), str(backup_path)]
            fallback.append(f"sudo cp {shlex.quote(str(rules_path))} {shlex.quote(str(backup_path))}")
            proc = run_cmd(cmd, timeout=10)
            if proc.returncode != 0:
                msg = proc.stderr.strip() or proc.stdout.strip() or "backup failed"
                return RuleApplyResult(False, f"backup failed: {msg}", fallback)

        cmd = ["sudo", "-n", "cp", tmp_path, str(rules_path)]
        fallback.append(f"sudo cp {shlex.quote(tmp_path)} {shlex.quote(str(rules_path))}")
        proc = run_cmd(cmd, timeout=10)
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or "copy failed"
            return RuleApplyResult(False, f"apply failed: {msg}", fallback)

        ok_reload, reload_msg = refresh_udev_rules(auto_prompt=False)
        if not ok_reload:
            fallback.extend(reload_msg.splitlines())
            return RuleApplyResult(False, "rules copied, but udev refresh failed", fallback)

        return RuleApplyResult(True, f"rules applied to system path: {rules_path}", fallback)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def refresh_udev_rules(auto_prompt: bool = True) -> Tuple[bool, str]:
    ok_sudo, sudo_msg = ensure_sudo_ready(auto_prompt=auto_prompt)
    if not ok_sudo:
        manual = [
            "sudo -v",
            "sudo udevadm control --reload-rules",
            "sudo udevadm trigger",
        ]
        return False, sudo_msg + "\n" + "\n".join(manual)

    commands = [
        ["sudo", "-n", "udevadm", "control", "--reload-rules"],
        ["sudo", "-n", "udevadm", "trigger"],
    ]
    fallback_lines = [
        "sudo udevadm control --reload-rules",
        "sudo udevadm trigger",
    ]

    for cmd in commands:
        proc = run_cmd(cmd, timeout=12)
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or f"failed: {' '.join(cmd)}"
            return False, msg + "\n" + "\n".join(fallback_lines)

    return True, "udev rules reloaded and triggered"

def sync_config_camera_map(config_path: Path, role_names: List[str]) -> Tuple[bool, str]:
    data = read_yaml(config_path)
    existing = data.get("camera_name_map", {})
    if not isinstance(existing, dict):
        existing = {}

    default_fov_h = float(data.get("default_fov_h", 60.0) or 60.0)
    default_fov_v = float(data.get("default_fov_v", 45.0) or 45.0)

    new_map = {}
    for role in role_names:
        val = existing.get(role)
        if isinstance(val, dict):
            yaw = float(val.get("yaw", 0.0) or 0.0)
            fov_h = float(val.get("fov_h", default_fov_h) or default_fov_h)
            fov_v = float(val.get("fov_v", default_fov_v) or default_fov_v)
            new_map[role] = {"yaw": yaw, "fov_h": fov_h, "fov_v": fov_v}
        elif isinstance(val, (int, float)):
            new_map[role] = {"yaw": float(val), "fov_h": default_fov_h, "fov_v": default_fov_v}
        else:
            new_map[role] = {"yaw": 0.0, "fov_h": default_fov_h, "fov_v": default_fov_v}

    ok, msg = write_camera_name_map_only(config_path, new_map)
    if not ok:
        return False, msg
    return True, f"updated camera_name_map in {config_path}"


def print_check_report(config_path: Path) -> int:
    print("========== Perception Camera Check ==========")
    print(f"Config: {config_path}")

    camera_map = get_camera_name_map(config_path)
    print("\nConfigured cameras (camera_name_map):")
    if not camera_map:
        print("  (none)")
    else:
        for name, value in camera_map.items():
            print(f"  - {name}: {value}")

    print("\n/dev/video*:")
    infos = scan_video_devices(test_open=True)
    if not infos:
        print("  (none)")
    for info in infos:
        print(
            f"  - {info.path} -> {info.realpath} | open={info.open_ok} frame={info.frame_ok} "
            f"{info.width}x{info.height}@{info.fps:.1f} | {info.id_vendor_id}:{info.id_model_id}"
        )

    print("\n/dev/camera* symlinks:")
    links = list_camera_symlinks()
    if not links:
        print("  (none)")
    for k, v in links.items():
        print(f"  - {k} -> {v}")

    print("\nExisting rules:")
    text = get_existing_rules_text()
    if text:
        print(text)
    else:
        print(f"  (missing or unreadable: {DEFAULT_RULES_PATH})")

    print("========== End ==========")
    return 0


def round_robin_preview(bindings: Dict[str, str], interval_sec: float, stop_event: threading.Event, stats: Dict[str, Dict[str, float]]) -> None:
    open_roles: List[str] = []
    caps: Dict[str, cv2.VideoCapture] = {}

    for role, device in bindings.items():
        if not device:
            continue
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            caps[role] = cap
            open_roles.append(role)
            stats.setdefault(role, {"frames": 0.0, "fails": 0.0, "last_ts": 0.0})
        else:
            stats.setdefault(role, {"frames": 0.0, "fails": 1.0, "last_ts": 0.0})

    if not open_roles:
        return

    idx = 0
    last_fps_tick = time.time()
    fps_counter = 0

    while not stop_event.is_set():
        role = open_roles[idx]
        idx = (idx + 1) % len(open_roles)

        cap = caps[role]
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            fps_counter += 1
            stats[role]["frames"] += 1
            stats[role]["last_ts"] = time.time()

            now = time.time()
            elapsed = max(now - last_fps_tick, 1e-6)
            fps = fps_counter / elapsed
            if elapsed >= 1.0:
                last_fps_tick = now
                fps_counter = 0

            text = f"{role} | RR interval={int(interval_sec*1000)}ms | FPS~{fps:.1f}"
            cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"RR Preview - {role}", frame)
        else:
            stats[role]["fails"] += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

        time.sleep(interval_sec)

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


class PerceptionCameraManagerUI:
    def __init__(self, config_path: Path):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("感知相机管理器")
        self.root.geometry("1200x760")

        self.config_path_var = tk.StringVar(value=str(config_path))
        self.enable_count_var = tk.IntVar(value=len(ROLE_TEMPLATES))
        self.role_vars: Dict[str, tk.StringVar] = {
            role_name: tk.StringVar(value="") for _, role_name in ROLE_TEMPLATES
        }
        self.role_enable_vars: Dict[str, tk.BooleanVar] = {
            role_name: tk.BooleanVar(value=True) for _, role_name in ROLE_TEMPLATES
        }
        self._init_roles_from_config_done = False

        self.devices: List[DeviceInfo] = []
        self.generated_rules: str = ""
        self.preview_stop_event = threading.Event()
        self.preview_thread: Optional[threading.Thread] = None
        self.preview_stats: Dict[str, Dict[str, float]] = {}

        self._build_layout()
        self._sync_role_ui_state()
        self.scan_devices()

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="配置文件:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.config_path_var, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="读取配置", command=self.load_config_info).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新设备", command=self.scan_devices).pack(side=tk.LEFT, padx=4)

        center = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        center.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left_panel = ttk.Frame(center)
        right_panel = ttk.Frame(center)
        center.add(left_panel, weight=3)
        center.add(right_panel, weight=2)

        ttk.Label(left_panel, text="视频设备列表").pack(anchor=tk.W)
        columns = ("path", "real", "id_path", "open", "frame", "size")
        self.tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=12)
        for c, title, width in [
            ("path", "设备", 110),
            ("real", "真实路径", 120),
            ("id_path", "ID_PATH", 260),
            ("open", "可打开", 70),
            ("frame", "有帧", 60),
            ("size", "分辨率/FPS", 120),
        ]:
            self.tree.heading(c, text=title)
            self.tree.column(c, width=width, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=4)

        role_frame = ttk.LabelFrame(right_panel, text="相机角色绑定（模板）")
        role_frame.pack(fill=tk.X, pady=4)

        ttk.Label(role_frame, text="启用相机数量").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        count_box = ttk.Combobox(
            role_frame,
            values=[str(i) for i in range(1, len(ROLE_TEMPLATES) + 1)],
            textvariable=self.enable_count_var,
            width=8,
            state="readonly",
        )
        count_box.grid(row=0, column=1, sticky=tk.W, padx=6, pady=4)
        count_box.bind("<<ComboboxSelected>>", lambda _e: self._on_enable_count_changed())

        self.role_boxes: Dict[str, ttk.Combobox] = {}
        for i, (role_label, role_name) in enumerate(ROLE_TEMPLATES, start=1):
            ttk.Checkbutton(
                role_frame,
                text="启用",
                variable=self.role_enable_vars[role_name],
                command=self._on_role_toggle,
            ).grid(row=i, column=0, sticky=tk.W, padx=6, pady=4)
            ttk.Label(role_frame, text=f"{role_label} ({role_name})").grid(row=i, column=1, sticky=tk.W, padx=6, pady=4)
            box = ttk.Combobox(role_frame, textvariable=self.role_vars[role_name], width=32, state="readonly")
            box.grid(row=i, column=2, sticky=tk.W, padx=6, pady=4)
            self.role_boxes[role_name] = box

        btn_frame = ttk.Frame(right_panel)
        btn_frame.pack(fill=tk.X, pady=4)

        ttk.Button(btn_frame, text="自动分配", command=self.auto_assign).grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="生成规则", command=self.generate_rules).grid(row=0, column=1, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="应用规则 (sudo)", command=self.apply_rules).grid(row=0, column=2, padx=4, pady=4, sticky=tk.W)

        ttk.Button(btn_frame, text="查看已有规则", command=self.show_existing_rules).grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="刷新规则", command=self.refresh_rules).grid(row=1, column=1, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="同步配置", command=self.sync_config).grid(row=1, column=2, padx=4, pady=4, sticky=tk.W)

        ttk.Button(btn_frame, text="开始轮询预览", command=self.start_preview).grid(row=2, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="停止预览", command=self.stop_preview).grid(row=2, column=1, padx=4, pady=4, sticky=tk.W)
        ttk.Button(btn_frame, text="导出报告", command=self.save_report).grid(row=2, column=2, padx=4, pady=4, sticky=tk.W)

        cmap_frame = ttk.LabelFrame(right_panel, text="配置中的 camera_name_map")
        cmap_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.cmap_text = tk.Text(cmap_frame, height=12)
        self.cmap_text.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(self.root, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _sync_role_ui_state(self) -> None:
        for _, role_name in ROLE_TEMPLATES:
            state = "readonly" if self.role_enable_vars[role_name].get() else "disabled"
            self.role_boxes[role_name]["state"] = state

    def _set_enable_count(self, count: int) -> None:
        count = max(1, min(len(ROLE_TEMPLATES), int(count)))
        self.enable_count_var.set(count)
        for i, (_, role_name) in enumerate(ROLE_TEMPLATES):
            self.role_enable_vars[role_name].set(i < count)
        self._sync_role_ui_state()

    def _on_enable_count_changed(self) -> None:
        self._set_enable_count(int(self.enable_count_var.get()))
        self.log(f"已设置启用相机数量: {self.enable_count_var.get()}")

    def _on_role_toggle(self) -> None:
        enabled_count = sum(1 for _, role_name in ROLE_TEMPLATES if self.role_enable_vars[role_name].get())
        if enabled_count <= 0:
            first_role = ROLE_TEMPLATES[0][1]
            self.role_enable_vars[first_role].set(True)
            enabled_count = 1
        self.enable_count_var.set(enabled_count)
        self._sync_role_ui_state()

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(self.tk.END, f"[{ts}] {message}\n")
        self.log_text.see(self.tk.END)

    def _config_path(self) -> Path:
        return Path(self.config_path_var.get()).expanduser().resolve()

    def load_config_info(self) -> None:
        cfg = self._config_path()
        cmap = get_camera_name_map(cfg)
        self.cmap_text.delete("1.0", self.tk.END)
        if not cmap:
            self.cmap_text.insert(self.tk.END, "camera_name_map 为空或不存在\n")
            self.log("配置中缺少 camera_name_map")
            return

        if not self._init_roles_from_config_done:
            template_roles = {role_name for _, role_name in ROLE_TEMPLATES}
            enabled_roles = [name for name in cmap.keys() if name in template_roles]
            if enabled_roles:
                for _, role_name in ROLE_TEMPLATES:
                    self.role_enable_vars[role_name].set(role_name in enabled_roles)
                self.enable_count_var.set(max(1, min(len(enabled_roles), len(ROLE_TEMPLATES))))
                self._sync_role_ui_state()
            self._init_roles_from_config_done = True

        self.cmap_text.insert(self.tk.END, yaml.safe_dump({"camera_name_map": cmap}, sort_keys=False, allow_unicode=True))
        self.log(f"已读取配置: {cfg}")

    def scan_devices(self) -> None:
        self.devices = scan_video_devices(test_open=True)
        links = list_camera_symlinks()

        for item in self.tree.get_children():
            self.tree.delete(item)

        for d in self.devices:
            self.tree.insert(
                "",
                self.tk.END,
                values=(
                    d.path,
                    d.realpath,
                    d.id_path,
                    str(d.open_ok),
                    str(d.frame_ok),
                    f"{d.width}x{d.height}@{d.fps:.1f}",
                ),
            )

        options = [d.path for d in self.devices]
        for role_name, box in self.role_boxes.items():
            box["values"] = options
            if self.role_enable_vars[role_name].get():
                if self.role_vars[role_name].get() not in options:
                    self.role_vars[role_name].set(options[0] if options else "")
            else:
                self.role_vars[role_name].set("")

        self._sync_role_ui_state()
        self.log(f"设备扫描完成: video设备={len(self.devices)}, 软链接={len(links)}")
        if links:
            for k, v in links.items():
                self.log(f"  {k} -> {v}")

        self.load_config_info()

    def auto_assign(self) -> None:
        options = [d.path for d in self.devices]
        for _, role_name in ROLE_TEMPLATES:
            self.role_vars[role_name].set("")

        for idx, role_name in enumerate(self.enabled_roles()):
            if idx < len(options):
                self.role_vars[role_name].set(options[idx])

        self.log("已按 /dev/video* 顺序自动分配启用相机")

    def current_bindings(self) -> Dict[str, str]:
        return {role_name: self.role_vars[role_name].get().strip() for _, role_name in ROLE_TEMPLATES}

    def enabled_roles(self) -> List[str]:
        return [role_name for _, role_name in ROLE_TEMPLATES if self.role_enable_vars[role_name].get()]

    def active_bindings(self) -> Dict[str, str]:
        bindings = self.current_bindings()
        return {role: bindings[role] for role in self.enabled_roles() if bindings.get(role)}

    def selected_roles(self) -> List[str]:
        return self.enabled_roles()

    def generate_rules(self) -> None:
        bindings = self.active_bindings()
        if not bindings:
            self.log("未为已启用相机选择设备，无法生成规则")
            return

        self.generated_rules, warns = generate_rules_text(bindings)
        self.log("已生成 udev 规则预览")
        for w in warns:
            self.log(f"警告: {w}")

        self._show_text_window("生成的规则", self.generated_rules)

    def apply_rules(self) -> None:
        if not self.generated_rules:
            self.generate_rules()

        result = apply_rules_text(self.generated_rules)
        if result.ok:
            self.log(result.message)
            self.scan_devices()
            return

        self.log(f"错误: {result.message}")
        if result.fallback_commands:
            self.log("可手动执行命令:")
            for cmd in result.fallback_commands:
                self.log(f"  {cmd}")

        self._show_text_window("应用失败", result.message + "\n\n" + "\n".join(result.fallback_commands))

    def refresh_rules(self) -> None:
        ok, msg = refresh_udev_rules()
        if ok:
            self.log(msg)
        else:
            self.log(f"错误: {msg}")
            self._show_text_window("刷新失败", msg)
        self.scan_devices()

    def show_existing_rules(self) -> None:
        text = get_existing_rules_text(DEFAULT_RULES_PATH)
        if not text:
            text = f"未找到系统规则: {DEFAULT_RULES_PATH}"

        display = "系统规则路径: " + str(DEFAULT_RULES_PATH) + "\n\n" + text
        self._show_text_window("已有规则", display)
        self.log(f"已查询系统规则: {DEFAULT_RULES_PATH}")

    def sync_config(self) -> None:
        roles = self.selected_roles()
        if not roles:
            self.log("没有启用的角色可写回配置")
            return
        ok, msg = sync_config_camera_map(self._config_path(), roles)
        if ok:
            self.log(msg)
            self.load_config_info()
        else:
            self.log(f"错误: 同步配置失败: {msg}")

    def start_preview(self) -> None:
        if self.preview_thread and self.preview_thread.is_alive():
            self.log("预览已经在运行")
            return

        bindings = self.active_bindings()
        if not bindings:
            self.log("没有可用于预览的已启用相机")
            return

        self.preview_stop_event.clear()
        self.preview_stats = {}
        self.preview_thread = threading.Thread(
            target=round_robin_preview,
            args=(bindings, POLL_INTERVAL_SEC, self.preview_stop_event, self.preview_stats),
            daemon=True,
        )
        self.preview_thread.start()
        self.log(f"已启动轮询预览，间隔={int(POLL_INTERVAL_SEC*1000)}ms")

    def stop_preview(self) -> None:
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_stop_event.set()
            self.preview_thread.join(timeout=2.0)
            self.log("已停止预览")
        else:
            self.log("预览未运行")

    def save_report(self) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = REPORT_DIR / f"perception_camera_report_{now}.json"
        txt_path = REPORT_DIR / f"perception_camera_report_{now}.txt"

        links = list_camera_symlinks()
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_path": str(self._config_path()),
            "camera_name_map": get_camera_name_map(self._config_path()),
            "enabled_roles": self.enabled_roles(),
            "bindings": self.current_bindings(),
            "devices": [asdict(d) for d in self.devices],
            "symlinks": links,
            "existing_rules": get_existing_rules_text(),
            "generated_rules": self.generated_rules,
            "preview_stats": self.preview_stats,
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        lines = [
            f"timestamp: {report['timestamp']}",
            f"config_path: {report['config_path']}",
            "",
            "camera_name_map:",
            yaml.safe_dump({"camera_name_map": report["camera_name_map"]}, allow_unicode=True, sort_keys=False),
            "enabled_roles:",
        ]
        for role in report["enabled_roles"]:
            lines.append(f"  - {role}")

        lines.append("\nbindings:")
        for k, v in report["bindings"].items():
            lines.append(f"  {k}: {v}")

        lines.append("\npreview_stats:")
        for k, v in report["preview_stats"].items():
            lines.append(f"  {k}: {v}")

        lines.append("\nsymlinks:")
        for k, v in links.items():
            lines.append(f"  {k} -> {v}")

        lines.append("\nexisting_rules:\n")
        lines.append(report["existing_rules"])

        lines.append("\ngenerated_rules:\n")
        lines.append(report["generated_rules"])

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        self.log(f"已保存报告: {json_path}")
        self.log(f"已保存报告: {txt_path}")

    def _show_text_window(self, title: str, text: str) -> None:
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("980x600")
        t = tk.Text(win)
        t.pack(fill=tk.BOTH, expand=True)
        t.insert(tk.END, text)

    def run(self) -> None:
        self.root.mainloop()

def cli_preview(config_path: Path, duration_sec: int) -> int:
    cmap = get_camera_name_map(config_path)
    if not cmap:
        print("camera_name_map missing in config")
        return 1

    names = list(cmap.keys())[:3]
    bindings = {name: f"/dev/{name}" for name in names}

    stop_event = threading.Event()
    stats: Dict[str, Dict[str, float]] = {}
    t = threading.Thread(
        target=round_robin_preview,
        args=(bindings, POLL_INTERVAL_SEC, stop_event, stats),
        daemon=True,
    )
    t.start()

    try:
        if duration_sec > 0:
            time.sleep(duration_sec)
            stop_event.set()
        t.join()
    except KeyboardInterrupt:
        stop_event.set()
        t.join()

    print("Preview stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Perception camera manager")
    p.add_argument("mode", nargs="?", default="ui", choices=["ui", "check", "preview", "show-rules", "refresh-rules"])
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="yaml config path")
    p.add_argument("--duration", type=int, default=0, help="preview duration in seconds (0=until quit)")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()

    if args.mode == "check":
        return print_check_report(config_path)

    if args.mode == "show-rules":
        print(f"System rules path: {DEFAULT_RULES_PATH}")
        text = get_existing_rules_text(DEFAULT_RULES_PATH)
        if text:
            print(text)
            return 0
        print(f"No rules found: {DEFAULT_RULES_PATH}")
        return 1

    if args.mode == "refresh-rules":
        ok, msg = refresh_udev_rules()
        print(msg)
        return 0 if ok else 1

    if args.mode == "preview":
        return cli_preview(config_path, args.duration)

    # UI mode
    try:
        ui = PerceptionCameraManagerUI(config_path)
        ui.run()
        return 0
    except Exception as e:
        print(f"UI failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())