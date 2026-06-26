#!/bin/bash
# =============================================================
#  Kronos 云服务器一键部署脚本 (Linux)
#  用法: bash setup_server.sh [--no-nginx] [--port 7070]
# =============================================================
set -e

# ── 默认参数 ──────────────────────────────────────────────────
APP_PORT=7070
INSTALL_NGINX=true
DEPLOY_USER=$(whoami)
ACCESS_PASSWORD="0391"               # 访问密码（留空则不启用）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEBUI_DIR="$PROJECT_DIR/webui"
VENV_DIR="$PROJECT_DIR/.venv"
SERVICE_NAME="kronos"

# ── 解析命令行参数 ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-nginx)   INSTALL_NGINX=false; shift ;;
        --port)       APP_PORT="$2"; shift 2 ;;
        --user)       DEPLOY_USER="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 颜色输出 ──────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 检查系统 ──────────────────────────────────────────────────
[[ "$EUID" -eq 0 ]] || error "请用 root 或 sudo 运行此脚本"

echo ""
echo "=============================================="
echo "  Kronos 部署配置"
echo "  项目目录 : $PROJECT_DIR"
echo "  运行用户 : $DEPLOY_USER"
echo "  服务端口 : $APP_PORT"
echo "  安装Nginx: $INSTALL_NGINX"
echo "=============================================="
echo ""

# ── 安装系统依赖 ──────────────────────────────────────────────
info "安装系统依赖..."
apt-get update -qq
apt-get install -y -qq curl unzip bc

if $INSTALL_NGINX; then
    apt-get install -y -qq nginx
fi

# ── 检测并安装合适的 Python 版本 ──────────────────────────────
# Ubuntu 20.04 默认 Python 3.8，需要升级到 3.9+
_pick_python() {
    for py in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "$py" &>/dev/null; then
            local ver
            ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
                echo "$py"; return
            fi
        fi
    done
}

PYTHON3=$(_pick_python)
if [[ -z "$PYTHON3" ]]; then
    info "当前 Python 低于 3.9，尝试安装 python3.11..."
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-distutils
    PYTHON3="python3.11"
fi
apt-get install -y -qq "${PYTHON3}-venv" 2>/dev/null || apt-get install -y -qq python3-venv

PYTHON_VERSION=$("$PYTHON3" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "使用 Python $PYTHON_VERSION ($PYTHON3)"

# 安装 pip
if ! "$PYTHON3" -m pip --version &>/dev/null 2>&1; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$PYTHON3"
fi

# ── 创建/更新虚拟环境 ─────────────────────────────────────────
info "配置 Python 虚拟环境..."
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON3" -m venv "$VENV_DIR"
    info "已创建虚拟环境: $VENV_DIR"
else
    info "虚拟环境已存在，跳过创建"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

info "安装 Python 依赖（使用清华镜像源）..."
"$VENV_PIP" install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"$VENV_PIP" install -r "$PROJECT_DIR/webui/requirements.txt" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
info "依赖安装完成"

# ── 写入访问密码 ──────────────────────────────────────────────
if [[ -n "$ACCESS_PASSWORD" ]]; then
    info "配置访问密码..."
    CONFIG_FILE="$WEBUI_DIR/datasource_config.json"
    python3 - <<PYEOF
import json, os
path = "$CONFIG_FILE"
cfg = {}
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
cfg['access_password'] = "$ACCESS_PASSWORD"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False)
PYEOF
fi

# ── 配置文件权限 ──────────────────────────────────────────────
info "设置目录权限..."
chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$PROJECT_DIR"
chmod -R 755 "$PROJECT_DIR"
# 配置文件和缓存需要可写
chmod 664 "$WEBUI_DIR/datasource_config.json" 2>/dev/null || true
chmod 664 "$WEBUI_DIR/futures_symbols.json"   2>/dev/null || true
mkdir -p "$WEBUI_DIR/prediction_results"
chmod 775 "$WEBUI_DIR/prediction_results"

# ── 生成 systemd service 文件 ─────────────────────────────────
info "配置 systemd 服务..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Kronos 金融预测平台
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$WEBUI_DIR
ExecStart=$VENV_PYTHON $WEBUI_DIR/app.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/kronos/app.log
StandardError=append:/var/log/kronos/error.log

# 环境变量
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$PROJECT_DIR
Environment=KRONOS_PORT=$APP_PORT

[Install]
WantedBy=multi-user.target
EOF

# 创建日志目录
mkdir -p /var/log/kronos
chown "$DEPLOY_USER":"$DEPLOY_USER" /var/log/kronos

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
info "systemd 服务已配置: $SERVICE_FILE"

# ── 配置 Nginx ────────────────────────────────────────────────
if $INSTALL_NGINX; then
    info "配置 Nginx 反向代理..."
    NGINX_CONF="/etc/nginx/sites-available/kronos"

    cat > "$NGINX_CONF" <<EOF
server {
    listen 80;
    server_name _;          # 替换为你的域名或IP

    # 上传大小限制（预测可能传较大请求体）
    client_max_body_size 16M;

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

    # 静态资源缓存
    location ~* \.(js|css|png|jpg|ico|woff2?)$ {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_cache_valid 200 1d;
        add_header Cache-Control "public, max-age=86400";
    }
}
EOF

    # 启用站点
    ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/kronos
    rm -f /etc/nginx/sites-enabled/default  # 移除默认站点

    nginx -t && systemctl restart nginx
    info "Nginx 配置完成"
fi

# ── 启动服务 ──────────────────────────────────────────────────
info "启动 Kronos 服务..."
systemctl restart "$SERVICE_NAME"
sleep 3

if systemctl is-active --quiet "$SERVICE_NAME"; then
    info "✅ Kronos 服务启动成功！"
else
    error "服务启动失败，请查看日志: journalctl -u kronos -n 50"
fi

# ── 完成提示 ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo -e "  ${GREEN}部署完成！${NC}"
echo ""
if $INSTALL_NGINX; then
    echo "  访问地址 : http://<你的服务器IP>"
else
    echo "  访问地址 : http://<你的服务器IP>:${APP_PORT}"
fi
echo ""
echo "  常用命令:"
echo "    查看状态 : systemctl status kronos"
echo "    查看日志 : tail -f /var/log/kronos/app.log"
echo "    重启服务 : systemctl restart kronos"
echo "    停止服务 : systemctl stop kronos"
echo "=============================================="
