#!/usr/bin/env bash
set -u

# 看门狗启动脚本。
# 职责：在新终端中启动目标程序，卡死超时强杀，崩溃后退避重启，硬件未就绪时等待。
# 终端输出同时写入日志，方便排查串口/相机掉线等问题。

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# ── 路径与参数配置 ──
BIN_REL="./build/standard"
CFG_REL="configs/infantry_5.yaml"
LOG_FILE="${PROJECT_ROOT}/logs/infantry_5.log"
PID_FILE="${PROJECT_ROOT}/logs/infantry_5.pid"

# ── 看门狗行为参数 ──
RESTART_DELAY_SEC=2          # 正常重启延迟（秒）
RUN_TIMEOUT_SEC=60           # 单次运行超时（秒），卡死即杀；0=禁用
STARTUP_GRACE_SEC=5          # 首次启动前等待硬件就绪的时间（秒）
QUICK_FAIL_THRESHOLD_SEC=3   # 进程在 N 秒内退出视为快速失败
QUICK_FAIL_MAX_COUNT=5       # 连续快速失败 N 次后进入长等待
LONG_WAIT_SEC=30             # 快速失败过多后的长等待时间（秒）
DEVICE_WAIT_TIMEOUT_SEC=30   # 等待硬件设备的超时（秒），0=不等待

# 可选：需要等待的设备节点，空格分隔。
# 示例：DEVICES="/dev/ttyACM0 /dev/ttyUSB0 /dev/video0"
DEVICES="${DEVICES:-}"

BIN_PATH="${PROJECT_ROOT}/${BIN_REL}"
CFG_PATH="${PROJECT_ROOT}/${CFG_REL}"

mkdir -p "${PROJECT_ROOT}/logs"

# ── 弹窗自启动：非终端环境下在新 gnome-terminal 中运行自己 ──
if [[ ! -t 0 ]] || [[ "${IN_NEW_TERMINAL:-}" != "1" ]]; then
    export IN_NEW_TERMINAL=1
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal -- bash -c "cd ${PROJECT_ROOT} && ./scripts/auto_start.sh $*; exec bash"
        exit 0
    fi
    echo "[$(date -Is)] WARNING: gnome-terminal not found, running in current tty" >> "${LOG_FILE}"
fi

ACTION="${1:-start}"
CHILD_PID=""

# ── 工具函数 ──

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    read -r pid < "${PID_FILE}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "${pid}"
      return 0
    fi
  fi
  return 1
}

stop_watchdog() {
  if pid=$(is_running); then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "[$(date -Is)] launcher stop" >> "${LOG_FILE}"
    echo "已停止，PID=${pid}"
  else
    echo "未在运行"
  fi
}

status_watchdog() {
  if pid=$(is_running); then
    echo "运行中，PID=${pid}"
    return 0
  fi
  echo "未在运行"
  return 1
}

# 等待硬件设备就绪。
wait_for_devices() {
  if [[ -z "${DEVICES}" ]]; then
    return 0
  fi

  local deadline=$(($(date +%s) + DEVICE_WAIT_TIMEOUT_SEC))
  local all_ok=true

  while true; do
    all_ok=true
    for dev in ${DEVICES}; do
      if [[ ! -e "${dev}" ]]; then
        all_ok=false
        break
      fi
    done
    if ${all_ok}; then
      echo "[$(date -Is)] all devices ready: ${DEVICES}" >> "${LOG_FILE}"
      return 0
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
      echo "[$(date -Is)] WARNING: device wait timeout, proceeding anyway" >> "${LOG_FILE}"
      return 1
    fi
    sleep 1
  done
}

# ── 命令分发 ──

case "${ACTION}" in
  start)
    if is_running >/dev/null; then
      echo "已在运行，PID=$(is_running)"
      exit 0
    fi
    ;;
  stop)
    stop_watchdog
    exit 0
    ;;
  status)
    status_watchdog
    exit $?
    ;;
  restart)
    stop_watchdog
    ;;
  *)
    echo "用法: $0 {start|stop|status|restart}"
    exit 2
    ;;
esac

echo $$ > "${PID_FILE}"

cleanup() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill "${CHILD_PID}" 2>/dev/null || true
    sleep 1
    kill -9 "${CHILD_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[$(date -Is)] launcher exit" >> "${LOG_FILE}"
  exit 0
}

trap cleanup INT TERM HUP

echo "[$(date -Is)] launcher start" >> "${LOG_FILE}"

# ── 首次启动前等待 ──
if [[ "${STARTUP_GRACE_SEC}" -gt 0 ]]; then
  echo "[$(date -Is)] startup grace: waiting ${STARTUP_GRACE_SEC}s for hardware to settle" >> "${LOG_FILE}"
  sleep "${STARTUP_GRACE_SEC}"
fi
wait_for_devices

# ── 主循环：启动 → 守护 → 重启 ──
cd "${PROJECT_ROOT}"

quick_fail_count=0
current_delay="${RESTART_DELAY_SEC}"

while true; do
  # 等待二进制就绪
  while [[ ! -x "${BIN_PATH}" ]]; do
    echo "[$(date -Is)] binary not found: ${BIN_PATH}" >> "${LOG_FILE}"
    sleep "${RESTART_DELAY_SEC}"
  done

  echo "[$(date -Is)] starting: ${BIN_PATH} ${CFG_PATH}" >> "${LOG_FILE}"

  START_TS=$(date +%s)

  # 使用进程替换 >(tee ...) 而非管道 | tee：
  # 管道中 $! 拿到的是 tee 的 PID，超时 kill 杀不到主程序。
  # 进程替换中 $! 是程序自身的 PID。
  "${BIN_PATH}" "${CFG_PATH}" > >(tee -a "${LOG_FILE}") 2>&1 &
  CHILD_PID=$!

  # 等待子进程，带超时检测。
  if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]]; then
    elapsed=0
    while kill -0 "${CHILD_PID}" 2>/dev/null; do
      sleep 1
      elapsed=$((elapsed + 1))
      if [[ ${elapsed} -ge "${RUN_TIMEOUT_SEC}" ]]; then
        echo "[$(date -Is)] TIMEOUT: killed after ${RUN_TIMEOUT_SEC}s" | tee -a "${LOG_FILE}"
        kill "${CHILD_PID}" 2>/dev/null || true
        sleep 1
        kill -9 "${CHILD_PID}" 2>/dev/null || true
        EXIT_CODE=124
        break
      fi
    done
    if [[ ${elapsed} -lt "${RUN_TIMEOUT_SEC}" ]]; then
      wait "${CHILD_PID}" 2>/dev/null
      EXIT_CODE=$?
    fi
  else
    wait "${CHILD_PID}" 2>/dev/null
    EXIT_CODE=$?
  fi

  END_TS=$(date +%s)
  RUN_DURATION=$((END_TS - START_TS))

  # ── 退避策略：快速失败累计 → 长等待 ──
  if [[ ${RUN_DURATION} -lt "${QUICK_FAIL_THRESHOLD_SEC}" ]]; then
    quick_fail_count=$((quick_fail_count + 1))
    echo "[$(date -Is)] quick fail #${quick_fail_count} (ran ${RUN_DURATION}s, exit=${EXIT_CODE})" | tee -a "${LOG_FILE}"

    if [[ ${quick_fail_count} -ge "${QUICK_FAIL_MAX_COUNT}" ]]; then
      echo "[$(date -Is)] too many quick fails, waiting ${LONG_WAIT_SEC}s for hardware recovery" | tee -a "${LOG_FILE}"
      sleep "${LONG_WAIT_SEC}"
      wait_for_devices
      quick_fail_count=0
      current_delay="${RESTART_DELAY_SEC}"
      continue
    fi

    # 指数退避：1s → 2s → 4s → 8s → 16s → max 30s
    current_delay=$((current_delay * 2))
    if [[ ${current_delay} -gt 30 ]]; then
      current_delay=30
    fi
  else
    # 正常运行超过阈值后退出，属于正常重启。
    quick_fail_count=0
    current_delay="${RESTART_DELAY_SEC}"
  fi

  if [[ ${EXIT_CODE} -eq 124 || ${EXIT_CODE} -eq 137 ]]; then
    echo "[$(date -Is)] timeout (${RUN_TIMEOUT_SEC}s), restarting in ${current_delay}s" | tee -a "${LOG_FILE}"
  else
    echo "[$(date -Is)] exit=${EXIT_CODE} after ${RUN_DURATION}s, restarting in ${current_delay}s" | tee -a "${LOG_FILE}"
  fi

  sleep "${current_delay}"

  # 长时间等待后重新检查硬件设备。
  if [[ ${current_delay} -ge 30 ]]; then
    wait_for_devices
  fi
done
