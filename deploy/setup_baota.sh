#!/bin/bash
# =============================================================
#  Kronos 宝塔面板一键部署脚本
#  在宝塔终端中运行:
#    bash <(curl -s 你的脚本地址)
#  或上传后运行:
#    bash /www/wwwroot/kronos/deploy/setup_baota.sh
# =============================================================
set -e

# ── 配置项（按需修改）────────────────────────────────────────
PROJECT_DIR="/www/wwwroot/kronos"   # 项目安装目录
APP_PORT=7070                        # Flask 监听端口
APP_NAME="kronos"                    # 进程/站点名称
GIT_URL=""                           # Git 仓库地址（留空则跳过克隆）
ACCESS_PASSWORD="0391"               # 访问密码（留空则不启用）
# ─────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[[ "$EUID" -eq 0 ]] || error "请用 root 用户运行"

echo ""
echo "=============================================="
echo "  Kronos 宝塔一键部署"
echo "  安装目录: $PROJECT_DIR"
echo "  服务端口: $APP_PORT"
echo "=============================================="
echo ""

# ── 1. 拉取项目代码 ───────────────────────────────────────────
if [[ -n "$GIT_URL" ]]; then
    info "克隆项目..."
    if [[ -d "$PROJECT_DIR/.git" ]]; then
        git -C "$PROJECT_DIR" pull
    else
        git clone "$GIT_URL" "$PROJECT_DIR"
    fi
elif [[ ! -d "$PROJECT_DIR/webui" ]]; then
    error "未找到项目目录 $PROJECT_DIR/webui，请先上传项目文件或配置 GIT_URL"
fi

WEBUI_DIR="$PROJECT_DIR/webui"
VENV_DIR="$PROJECT_DIR/.venv"

# ── 2. 找宝塔 Python3 ────────────────────────────────────────
info "查找 Python3..."
# 优先用宝塔安装的 Python
PYTHON3=$(find /www/server/pyenv/versions -name "python3" -type f 2>/dev/null | sort -V | tail -1)
if [[ -z "$PYTHON3" ]]; then
    PYTHON3=$(which python3 2>/dev/null || which python 2>/dev/null)
fi
[[ -z "$PYTHON3" ]] && error "未找到 Python3，请在宝塔软件商店安装 Python 项目管理器"

PY_VER=$("$PYTHON3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "使用 Python $PY_VER: $PYTHON3"
[[ $(echo "$PY_VER >= 3.9" | bc -l) -eq 1 ]] || error "需要 Python 3.9+，当前版本 $PY_VER"

# ── 3. 创建虚拟环境 ───────────────────────────────────────────
info "配置虚拟环境..."
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON3" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

info "安装项目依赖（首次较慢，请耐心等待）..."
"$VENV_PIP" install --upgrade pip -q
"$VENV_PIP" install -r "$WEBUI_DIR/requirements.txt" -q
info "依赖安装完成"

# ── 4. 在系统 Python 上额外装 tqsdk（供子进程调用）─────────────
info "在系统 Python 上安装 tqsdk..."
"$PYTHON3" -m pip install tqsdk pandas -q && info "tqsdk 安装成功" \
    || warn "tqsdk 安装失败，期货数据功能暂不可用，可稍后手动运行: $PYTHON3 -m pip install tqsdk"

# ── 5. 写入访问密码到配置文件 ─────────────────────────────────
if [[ -n "$ACCESS_PASSWORD" ]]; then
    info "配置访问密码..."
    CONFIG_FILE="$WEBUI_DIR/datasource_config.json"
    if [[ -f "$CONFIG_FILE" ]]; then
        # 用 Python 更新 JSON，避免覆盖已有配置
        "$PYTHON3" - <<PYEOF
import json, os
path = "$CONFIG_FILE"
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['access_password'] = "$ACCESS_PASSWORD"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False)
print("访问密码已写入配置")
PYEOF
    else
        echo "{\"access_password\": \"$ACCESS_PASSWORD\"}" > "$CONFIG_FILE"
        info "配置文件已创建，访问密码已设置"
    fi
fi

# ── 6. 设置目录权限 ───────────────────────────────────────────
info "设置权限..."
chown -R www:www "$PROJECT_DIR" 2>/dev/null || chown -R root:root "$PROJECT_DIR"
chmod -R 755 "$PROJECT_DIR"
chmod 664 "$WEBUI_DIR/datasource_config.json" 2>/dev/null || true
mkdir -p "$WEBUI_DIR/prediction_results"
chmod 775 "$WEBUI_DIR/prediction_results"

# ── 6. 配置 Supervisor 守护进程 ───────────────────────────────
info "配置 Supervisor..."

# 查找宝塔 supervisor 配置目录
if [[ -d "/www/server/panel/plugin/supervisor" ]]; then
    SUPERVISOR_CONF_DIR="/www/server/panel/plugin/supervisor/conf"
elif [[ -d "/etc/supervisor/conf.d" ]]; then
    SUPERVISOR_CONF_DIR="/etc/supervisor/conf.d"
else
    SUPERVISOR_CONF_DIR="/etc/supervisord.d"
fi
mkdir -p "$SUPERVISOR_CONF_DIR"

cat > "$SUPERVISOR_CONF_DIR/${APP_NAME}.conf" <<EOF
[program:${APP_NAME}]
directory=${WEBUI_DIR}
command=${VENV_PYTHON} ${WEBUI_DIR}/app.py
autostart=true
autorestart=true
startsecs=5
startretries=3
stdout_logfile=/var/log/${APP_NAME}/app.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
stderr_logfile=/var/log/${APP_NAME}/error.log
stderr_logfile_maxbytes=10MB
environment=PYTHONUNBUFFERED="1",PYTHONPATH="${PROJECT_DIR}",KRONOS_PORT="${APP_PORT}"
user=root
EOF

mkdir -p "/var/log/${APP_NAME}"

# 重载 supervisor
if command -v supervisorctl &>/dev/null; then
    supervisorctl reread && supervisorctl update
    supervisorctl restart "${APP_NAME}" 2>/dev/null || supervisorctl start "${APP_NAME}"
    info "Supervisor 守护进程已启动"
else
    warn "未找到 supervisorctl，请在宝塔面板 → 软件商店 安装 Supervisor"
fi

# ── 7. 配置 Nginx 反向代理 ────────────────────────────────────
info "配置 Nginx..."

# 查找宝塔 nginx vhost 目录
NGINX_VHOST_DIR=""
for dir in "/www/server/nginx/conf/vhost" "/www/server/panel/vhost/nginx"; do
    [[ -d "$dir" ]] && NGINX_VHOST_DIR="$dir" && break
done

if [[ -n "$NGINX_VHOST_DIR" ]]; then
    cat > "${NGINX_VHOST_DIR}/${APP_NAME}.conf" <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 16M;
    access_log  /var/log/${APP_NAME}/nginx_access.log;
    error_log   /var/log/${APP_NAME}/nginx_error.log;

    location / {
        proxy_pass         http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
EOF
    # 重载 nginx
    if command -v nginx &>/dev/null; then
        nginx -t && nginx -s reload && info "Nginx 配置完成"
    fi
else
    warn "未找到 Nginx vhost 目录，请在宝塔面板手动配置反向代理 → 目标: http://127.0.0.1:${APP_PORT}"
fi

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo -e "  ${GREEN}部署完成！${NC}"
echo ""
echo "  访问地址 : http://$(curl -s ifconfig.me 2>/dev/null || echo '你的服务器IP')"
echo "  直接访问 : http://$(curl -s ifconfig.me 2>/dev/null || echo '你的服务器IP'):${APP_PORT}"
echo ""
echo "  常用命令:"
echo "    查看状态 : supervisorctl status ${APP_NAME}"
echo "    查看日志 : tail -f /var/log/${APP_NAME}/app.log"
echo "    重启服务 : supervisorctl restart ${APP_NAME}"
echo "=============================================="
