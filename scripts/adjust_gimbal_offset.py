#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R_camera2gimbal / t_camera2gimbal 偏移调整工具 - 完整物理实战版

功能：
- 剥离基础坐标系差异 (-90, 0, -90)
- 内置实战调参口诀与方向指引
- 同时支持调整相机安装的物理旋转偏差（度）与平移偏差（毫米）
- 自动转换单位并生成兼容底层 YAML 的格式
"""

import os
import sys
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    print("错误：缺少 tkinter 模块")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("错误：缺少 pyyaml 模块")
    sys.exit(1)

# ==================== 核心：理想坐标系转换矩阵 ====================
# OpenCV相机(Z前,X右,Y下) 到 云台FLU(X前,Y左,Z上) 的理想转换矩阵
R_IDEAL = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0]
], dtype=float)

# ==================== 数学与配置解析工具 ====================

def wrap_angle_deg(angle: float) -> float:
    """将角度循环限制在 [-180, 180] 范围内"""
    return ((angle + 180.0) % 360.0) - 180.0

def euler_to_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    
    R = Rz @ Ry @ Rx
    R[np.abs(R) < 1e-10] = 0.0
    return R

def matrix_to_euler(R: np.ndarray) -> tuple:
    M = np.asarray(R, dtype=float)
    sin_pitch = np.clip(-M[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)
    
    if abs(cos_pitch) < 1e-6:
        yaw = np.arctan2(-M[0, 1], M[1, 1])
        roll = 0.0
    else:
        yaw = np.arctan2(M[1, 0], M[0, 0])
        roll = np.arctan2(M[2, 1] / cos_pitch, M[2, 2] / cos_pitch)
    
    return wrap_angle_deg(np.rad2deg(yaw)), wrap_angle_deg(np.rad2deg(pitch)), wrap_angle_deg(np.rad2deg(roll))

def get_deviation_euler(R_cam2gimbal: np.ndarray) -> tuple:
    """提取纯粹的物理偏差欧拉角"""
    R_error = R_cam2gimbal @ R_IDEAL.T
    return matrix_to_euler(R_error)

def build_matrix_from_deviation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """根据物理偏差角，重新构建最终的外参矩阵"""
    R_error = euler_to_matrix(np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), np.deg2rad(roll_deg))
    R_cam2gimbal = R_error @ R_IDEAL
    R_cam2gimbal[np.abs(R_cam2gimbal) < 1e-10] = 0.0
    return R_cam2gimbal

def format_vector_flat(V: np.ndarray) -> str:
    return ', '.join(f"{x:g}" for x in V.flatten())

def load_config(path: str) -> tuple:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None: raise ValueError("配置文件为空")
    
    R_cam = data.get('R_camera2gimbal') or data.get('R_original')
    if R_cam is None: raise ValueError("未找到 R_camera2gimbal")
    
    t_cam = data.get('t_camera2gimbal', [0.0, 0.0, 0.0])
    
    return np.array(R_cam, dtype=float).reshape(3, 3), np.array(t_cam, dtype=float)

# ==================== GUI 界面 ====================

class GimbalOffsetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("相机外参物理校准工具 (旋转 + 平移)")
        self.root.geometry("900x950")
        
        self.R_original = None
        self.t_original = None
        self.last_yaml = None
        self.create_widgets()
    
    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # --- 0. 调参口诀表 ---
        guide_frame = ttk.LabelFrame(main, text="🌟 实战调参口诀 (基于 FLU 云台系)", padding=10)
        guide_frame.pack(fill=tk.X, pady=(0, 10))
        
        guide_text = """【旋转】 左偏 ➔ Yaw 加正(+) | 上偏 ➔ Pitch 加正(+) | 画面左歪 ➔ Roll 加正(+)
【平移】 如果算法准星总是存在固定间距的平移误差(无法通过旋转对齐)，则需要调整平移：
        ➔ X 轴：相机靠前为正(+) | Y 轴：相机偏左为正(+) | Z 轴：相机偏上为正(+)"""
        ttk.Label(guide_frame, text=guide_text, font=("Microsoft YaHei", 10, "bold"), foreground="#D35400", justify=tk.LEFT).pack(anchor=tk.W)

        # --- 1. 配置文件 ---
        frame = ttk.LabelFrame(main, text="1. 加载配置文件", padding=5)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        self.config_entry = ttk.Entry(frame)
        self.config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.config_entry.insert(0, "../configs/infantry.yaml")
        ttk.Button(frame, text="浏览", command=self.browse).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="解析当前偏差", command=self.load_config).pack(side=tk.LEFT)
        
        # 当前偏差显示
        info_frame = ttk.Frame(main)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.info_label_rot = ttk.Label(info_frame, text="当前安装旋转: (请先加载配置)", font=("Microsoft YaHei", 11, "bold"), foreground="blue")
        self.info_label_rot.pack(anchor=tk.W)
        self.info_label_trans = ttk.Label(info_frame, text="当前安装平移: (请先加载配置)", font=("Microsoft YaHei", 11, "bold"), foreground="purple")
        self.info_label_trans.pack(anchor=tk.W, pady=(5, 0))
        
        # --- 2. 模式选择与输入 ---
        frame = ttk.LabelFrame(main, text="2. 调整物理外参", padding=5)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(frame, text="微调模式 (在原偏差上加减)", variable=self.mode_var,
                       value="manual", command=self.switch_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(frame, text="绝对模式 (直接指定新数值)", variable=self.mode_var,
                       value="target", command=self.switch_mode).pack(side=tk.LEFT)
        
        # 手动偏移输入容器
        self.manual_frame = ttk.Frame(frame, padding=5)
        
        # 旋转微调部分
        ttk.Label(self.manual_frame, text="【旋转调整 (单位：度°)】", foreground="blue", font=("", 10, "bold")).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))
        rot_tips = [
            ("Yaw 微调 (左右):", "yaw", "【填 +】: 纠正弹着点偏右", "【填 -】: 纠正弹着点偏左"), 
            ("Pitch 微调 (上下):", "pitch", "【填 +】: 纠正弹着点偏上", "【填 -】: 纠正弹着点偏下"), 
            ("Roll 微调 (旋转):", "roll", "【填 +】: 纠正画面向左倾斜(\\)", "【填 -】: 纠正画面向右倾斜(/)")
        ]
        
        for i, (label, key, tip_pos, tip_neg) in enumerate(rot_tips, start=1):
            ttk.Label(self.manual_frame, text=label, font=("", 10, "bold")).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(self.manual_frame, width=8, font=("", 10))
            entry.grid(row=i, column=1, sticky=tk.W, padx=10)
            entry.insert(0, "0.0")
            setattr(self, f"entry_{key}", entry)
            ttk.Label(self.manual_frame, text=tip_pos, foreground="green").grid(row=i, column=2, sticky=tk.W, padx=5)
            ttk.Label(self.manual_frame, text=tip_neg, foreground="red").grid(row=i, column=3, sticky=tk.W, padx=5)

        # 平移微调部分
        ttk.Label(self.manual_frame, text="【平移调整 (单位：毫米 mm)】", foreground="purple", font=("", 10, "bold")).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(15, 5))
        trans_tips = [
            ("X 微调 (前后):", "tx_off", "【填 +】: 相机实际更靠前", "【填 -】: 相机实际更靠后"), 
            ("Y 微调 (左右):", "ty_off", "【填 +】: 相机实际更靠左", "【填 -】: 相机实际更靠右"), 
            ("Z 微调 (上下):", "tz_off", "【填 +】: 相机实际更靠上", "【填 -】: 相机实际更靠下")
        ]
        
        for i, (label, key, tip_pos, tip_neg) in enumerate(trans_tips, start=5):
            ttk.Label(self.manual_frame, text=label, font=("", 10, "bold")).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(self.manual_frame, width=8, font=("", 10))
            entry.grid(row=i, column=1, sticky=tk.W, padx=10)
            entry.insert(0, "0.0")
            setattr(self, f"entry_{key}", entry)
            ttk.Label(self.manual_frame, text=tip_pos, foreground="green").grid(row=i, column=2, sticky=tk.W, padx=5)
            ttk.Label(self.manual_frame, text=tip_neg, foreground="red").grid(row=i, column=3, sticky=tk.W, padx=5)

        # 绝对目标输入容器
        self.target_frame = ttk.Frame(frame, padding=5)
        ttk.Label(self.target_frame, text="【绝对旋转偏角 (度°)】", foreground="blue", font=("", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        for i, (label, key) in enumerate([("目标 Yaw 偏角:", "ty"), ("目标 Pitch 偏角:", "tp"), ("目标 Roll 偏角:", "tr")], start=1):
            ttk.Label(self.target_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(self.target_frame, width=10)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            setattr(self, f"entry_{key}", entry)
            ttk.Label(self.target_frame, text="(留空则保持原偏差不变)", foreground="gray").grid(row=i, column=2, sticky=tk.W)
            
        ttk.Label(self.target_frame, text="【绝对平移位置 (毫米 mm)】", foreground="purple", font=("", 10, "bold")).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(15, 5))
        for i, (label, key) in enumerate([("目标 X 位置:", "tgt_tx"), ("目标 Y 位置:", "tgt_ty"), ("目标 Z 位置:", "tgt_tz")], start=5):
            ttk.Label(self.target_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(self.target_frame, width=10)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5)
            setattr(self, f"entry_{key}", entry)
            ttk.Label(self.target_frame, text="(留空则保持原位置不变)", foreground="gray").grid(row=i, column=2, sticky=tk.W)

        self.switch_mode()
        
        # --- 按钮 ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="⚡ 生成新外参", command=self.calculate, style="Accent.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清空输入", command=self.reset).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="📋 一键复制 YAML", command=self.copy_yaml).pack(side=tk.RIGHT, padx=2)
        
        # --- 3. 结果显示 ---
        frame = ttk.LabelFrame(main, text="3. 计算结果 (请将此段代码复制到 yaml 文件中)", padding=5)
        frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.result_text = tk.Text(frame, font=("Courier New", 10), height=14)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        self.status_var = tk.StringVar(value="等待加载配置...")
        ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(5, 0))
    
    def switch_mode(self):
        mode = self.mode_var.get()
        if mode == "manual":
            self.manual_frame.pack(fill=tk.X)
            self.target_frame.pack_forget()
        else:
            self.manual_frame.pack_forget()
            self.target_frame.pack(fill=tk.X)
    
    def browse(self):
        path = filedialog.askopenfilename(title="选择配置文件", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.config_entry.delete(0, tk.END)
            self.config_entry.insert(0, path)
            self.load_config()
    
    def load_config(self):
        path = self.config_entry.get().strip()
        try:
            self.R_original, self.t_original = load_config(path)
            y_err, p_err, r_err = get_deviation_euler(self.R_original)
            
            # YAML 中 t 是米，界面显示毫米
            tx_mm, ty_mm, tz_mm = self.t_original * 1000.0
            
            self.info_label_rot.config(text=f"当前安装旋转: Yaw={y_err:.2f}°, Pitch={p_err:.2f}°, Roll={r_err:.2f}°")
            self.info_label_trans.config(text=f"当前安装平移: X={tx_mm:.2f} mm, Y={ty_mm:.2f} mm, Z={tz_mm:.2f} mm")
            self.status_var.set(f"✅ 已成功解析配置: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"解析失败: {e}")
    
    def calculate(self):
        if self.R_original is None or self.t_original is None:
            messagebox.showwarning("警告", "请先加载配置文件！")
            return
        
        orig_y, orig_p, orig_r = get_deviation_euler(self.R_original)
        orig_tx, orig_ty, orig_tz = self.t_original * 1000.0  # 转为 mm 进行计算
        
        try:
            if self.mode_var.get() == "manual":
                # 获取旋转微调
                yaw_off = float(self.entry_yaw.get() or 0)
                pitch_off = float(self.entry_pitch.get() or 0)
                roll_off = float(self.entry_roll.get() or 0)
                # 获取平移微调
                tx_off = float(self.entry_tx_off.get() or 0)
                ty_off = float(self.entry_ty_off.get() or 0)
                tz_off = float(self.entry_tz_off.get() or 0)
                
                new_y = orig_y + yaw_off
                new_p = orig_p + pitch_off
                new_r = orig_r + roll_off
                
                new_tx = orig_tx + tx_off
                new_ty = orig_ty + ty_off
                new_tz = orig_tz + tz_off
                
                mode_desc = f"【微调模式】"
                
            else:
                # 获取旋转目标
                ty, tp, tr = self.entry_ty.get().strip(), self.entry_tp.get().strip(), self.entry_tr.get().strip()
                new_y = float(ty) if ty else orig_y
                new_p = float(tp) if tp else orig_p
                new_r = float(tr) if tr else orig_r
                
                # 获取平移目标
                t_tx, t_ty, t_tz = self.entry_tgt_tx.get().strip(), self.entry_tgt_ty.get().strip(), self.entry_tgt_tz.get().strip()
                new_tx = float(t_tx) if t_tx else orig_tx
                new_ty = float(t_ty) if t_ty else orig_ty
                new_tz = float(t_tz) if t_tz else orig_tz
                
                mode_desc = f"【绝对模式】"
            
            # 重构 R 矩阵
            new_y, new_p, new_r = wrap_angle_deg(new_y), wrap_angle_deg(new_p), wrap_angle_deg(new_r)
            R_corrected = build_matrix_from_deviation(new_y, new_p, new_r)
            
            # 重构 T 矩阵 (必须将计算的毫米换算回YAML底层使用的米)
            t_corrected = np.array([new_tx, new_ty, new_tz]) / 1000.0
            
            # ========== 界面打印反馈 ==========
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"--- 操作: {mode_desc} ---\n\n")
            
            self.result_text.insert(tk.END, "====== 旋转安装偏差变化对比 ======\n")
            self.result_text.insert(tk.END, f"  Yaw 偏角: {orig_y:8.2f}°  ==>  {new_y:8.2f}°  (变化 {new_y-orig_y:+.2f}°)\n")
            self.result_text.insert(tk.END, f"  Pitch 偏角: {orig_p:8.2f}°  ==>  {new_p:8.2f}°  (变化 {new_p-orig_p:+.2f}°)\n")
            self.result_text.insert(tk.END, f"  Roll 偏角: {orig_r:8.2f}°  ==>  {new_r:8.2f}°  (变化 {new_r-orig_r:+.2f}°)\n\n")
            
            self.result_text.insert(tk.END, "====== 平移安装位置变化对比 ======\n")
            self.result_text.insert(tk.END, f"  X 位置: {orig_tx:8.2f} mm  ==>  {new_tx:8.2f} mm  (变化 {new_tx-orig_tx:+.2f} mm)\n")
            self.result_text.insert(tk.END, f"  Y 位置: {orig_ty:8.2f} mm  ==>  {new_ty:8.2f} mm  (变化 {new_ty-orig_ty:+.2f} mm)\n")
            self.result_text.insert(tk.END, f"  Z 位置: {orig_tz:8.2f} mm  ==>  {new_tz:8.2f} mm  (变化 {new_tz-orig_tz:+.2f} mm)\n\n")
            
            yaml_out = f"R_camera2gimbal: [{format_vector_flat(R_corrected)}]\n"
            yaml_out += f"t_camera2gimbal: [{format_vector_flat(t_corrected)}]"
            self.result_text.insert(tk.END, "====== 复制下方代码替换到 yaml 中 (注意单位已自动转为米) ======\n")
            self.result_text.insert(tk.END, yaml_out + "\n")
            
            self.last_yaml = yaml_out
            self.status_var.set("✅ 计算完成！点击右侧按钮一键复制全部 YAML 数据")
            
        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {e}")
    
    def copy_yaml(self):
        if self.last_yaml:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_yaml)
            self.status_var.set("✅ 旋转矩阵与平移矩阵已成功复制到剪贴板！可以直接去粘贴了")
    
    def reset(self):
        for e in [self.entry_yaw, self.entry_pitch, self.entry_roll, self.entry_tx_off, self.entry_ty_off, self.entry_tz_off]:
            e.delete(0, tk.END); e.insert(0, "0.0")
        for e in [self.entry_ty, self.entry_tp, self.entry_tr, self.entry_tgt_tx, self.entry_tgt_ty, self.entry_tgt_tz]:
            e.delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    # 为计算按钮添加醒目样式
    if "Accent.TButton" not in style.theme_names():
        style.configure("Accent.TButton", font=("Microsoft YaHei", 10, "bold"), foreground="blue")
    GimbalOffsetGUI(root)
    root.mainloop()