#!/bin/bash
# =============================================================
#  Kronos 运维管理脚本
#  用法: bash manage.sh [start|stop|restart|status|log|update]
# =============================================================

SERVICE="kronos"
LOG_FILE="/var/log/kronos/app.log"
ERR_FILE="/var/log/kronos/error.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

case "$1" in
    start)
        echo -e "${GREEN}启动 Kronos...${NC}"
        systemctl start $SERVICE
        sleep 2
        systemctl is-active --quiet $SERVICE && echo -e "${GREEN}✅ 已启动${NC}" || echo -e "${RED}❌ 启动失败，查看日志: bash manage.sh log${NC}"
        ;;
    stop)
        echo -e "${YELLOW}停止 Kronos...${NC}"
        systemctl stop $SERVICE
        echo "已停止"
        ;;
    restart)
        echo -e "${YELLOW}重启 Kronos...${NC}"
        systemctl restart $SERVICE
        sleep 2
        systemctl is-active --quiet $SERVICE && echo -e "${GREEN}✅ 已重启${NC}" || echo -e "${RED}❌ 重启失败${NC}"
        ;;
    status)
        systemctl status $SERVICE --no-pager
        ;;
    log)
        echo -e "${GREEN}=== 应用日志 (Ctrl+C 退出) ===${NC}"
        tail -f "$LOG_FILE"
        ;;
    errlog)
        echo -e "${RED}=== 错误日志 (Ctrl+C 退出) ===${NC}"
        tail -f "$ERR_FILE"
        ;;
    update)
        # 拉取最新代码后重启（适合 git 部署）
        echo -e "${YELLOW}更新代码并重启...${NC}"
        cd "$PROJECT_DIR" || exit 1
        git pull
        VENV_PIP="$PROJECT_DIR/.venv/bin/pip"
        "$VENV_PIP" install -r "$PROJECT_DIR/webui/requirements.txt" -q
        systemctl restart $SERVICE
        sleep 2
        systemctl is-active --quiet $SERVICE && echo -e "${GREEN}✅ 更新完成${NC}" || echo -e "${RED}❌ 重启失败${NC}"
        ;;
    *)
        echo "用法: bash manage.sh [start|stop|restart|status|log|errlog|update]"
        echo ""
        echo "  start   - 启动服务"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  status  - 查看运行状态"
        echo "  log     - 实时查看应用日志"
        echo "  errlog  - 实时查看错误日志"
        echo "  update  - 拉取最新代码并重启"
        ;;
esac
